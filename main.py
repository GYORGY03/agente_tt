import os
import asyncio
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

GEMINI_API_KEY = "AIzaSyD8_B7f5QFkdSDrrQ49YJd5fgMwRYRdjzM"
POSTGRES_CONNECTION_STRING = "postgresql://postgres:laberinto@localhost:5432/labtech_bot"
QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = "e5362baf-c777-4d57-a609-6eaf1f9e87f6"
QDRANT_COLLECTION_1="documentos_gemini"
QDRANT_COLLECTION_2="documentos_tarifas"

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY no configurada. Instanciación final del LLM fallará sin ella.")


app = FastAPI(title="Chat Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://agent-tt.netlify.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "*"  
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    chat_id: str
    message: str

class ChatResponse(BaseModel):
    chat_id: str
    response: str


try:
    import asyncpg
except Exception: 
    asyncpg = None


class PostgresChatMemory:

    def __init__(self, dsn: str):
        if asyncpg is None:
            raise RuntimeError("`asyncpg` no está instalado. Instala asyncpg para usar PostgresChatMemory.")
        if not dsn:
            raise RuntimeError("POSTGRES_CONNECTION_STRING no configurada.")
        self._dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None

    async def init(self) -> None:
        if self._pool:
            print("[POSTGRES] Pool de conexiones ya inicializado")
            return
        
        print(f"[POSTGRES] Conectando a PostgreSQL...")
        print(f"[POSTGRES] DSN: {self._dsn[:50]}...")
        
        try:
            self._pool = await asyncpg.create_pool(self._dsn)
            print("[POSTGRES] ✅ Pool de conexiones creado exitosamente")
            
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages_web (
                        id SERIAL PRIMARY KEY,
                        chat_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )
                print("[POSTGRES] ✅ Tabla 'chat_messages_web' verificada/creada")
                

                result = await conn.fetchval("SELECT COUNT(*) FROM chat_messages_web")
                print(f"[POSTGRES] ✅ Mensajes en base de datos: {result}")
        except Exception as e:
            print(f"[POSTGRES] ❌ Error al conectar: {type(e).__name__}: {str(e)}")
            raise

    async def add_message(self, chat_id: str, role: str, content: str) -> None:
        if not self._pool:
            await self.init()
        
        print(f"[POSTGRES] Guardando mensaje - Chat: {chat_id}, Role: {role}")
        
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO chat_messages_web(chat_id, role, content) VALUES($1, $2, $3)",
                    chat_id,
                    role,
                    content,
                )
                print(f"[POSTGRES] ✅ Mensaje guardado exitosamente")
        except Exception as e:
            print(f"[POSTGRES] ❌ Error al guardar mensaje: {type(e).__name__}: {str(e)}")
            raise

    async def get_recent(self, chat_id: str, limit: int = 10) -> List[Dict[str, str]]:
        if not self._pool:
            await self.init()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT role, content, created_at FROM chat_messages_web WHERE chat_id=$1 ORDER BY created_at DESC LIMIT $2",
                chat_id,
                limit,
            )
        return [dict(row) for row in reversed(rows)]



try:
    from qdrant_client import QdrantClient
    from langchain_qdrant import QdrantVectorStore
    from langchain_core.documents import Document
except Exception:
    QdrantClient = None
    QdrantVectorStore = None
    Document = None


def init_qdrant_client() -> Optional[QdrantClient]:
    if QdrantClient is None:
        print("WARNING: qdrant-client o langchain no instalados; las herramientas RAG no estarán disponibles.")
        return None
    kwargs = {}
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    client = QdrantClient(url=QDRANT_URL, **kwargs)
    return client


def create_retrieval_tool_from_collection(collection_name: str, qdrant_client: QdrantClient, embeddings) -> Any:


    if QdrantVectorStore is None or Document is None:
    
        def missing_tool(query: str, metadata_filter: Optional[Dict] = None):
            return [{"page_content": "Qdrant no disponible: instala qdrant-client/langchain-qdrant"}]

        return missing_tool

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
        content_payload_key="text", 
    )

    def tool_sync(query: str, k: int = 3, metadata_filter: Optional[Dict] = None, score_threshold: float = 0.3) -> List[Any]:
        print(f"[QDRANT TOOL] Ejecutando búsqueda: query='{query[:50]}...', k={k}")
        
        try:

            search_k = k * 4
            results = vectorstore.similarity_search_with_score(query, k=search_k)
            print(f"[QDRANT TOOL] Resultados brutos obtenidos: {len(results)}")
            

            query_lower = query.lower()
            query_terms = set(query_lower.split())
            
            scored_docs = []
            for idx, (doc, vector_score) in enumerate(results):
                content = getattr(doc, 'page_content', '')
                
                if not content or len(content.strip()) < 20:
                    continue
                
                content_lower = content.lower()
                
  
                term_matches = sum(1 for term in query_terms if len(term) > 3 and term in content_lower)
                term_score = term_matches / max(len(query_terms), 1)
                
                
                has_substantive_text = len([w for w in content_lower.split() if len(w) > 5]) > 10
                text_quality_bonus = 0.1 if has_substantive_text else 0.0
                
           
                combined_score = (-vector_score if vector_score < 0 else vector_score) + (term_score * 0.3) + text_quality_bonus
                
                scored_docs.append({
                    'doc': doc,
                    'vector_score': vector_score,
                    'term_score': term_score,
                    'combined_score': combined_score,
                    'content': content
                })

            scored_docs.sort(key=lambda x: x['combined_score'], reverse=True)
            
            filtered_docs = []
            for idx, item in enumerate(scored_docs[:k]):
                doc = item['doc']
                print(f"[QDRANT TOOL] Doc {idx+1}: vector={item['vector_score']:.4f}, terms={item['term_score']:.2f}, combined={item['combined_score']:.4f}, len={len(item['content'])}, preview={item['content'][:80]}...")
                

                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata['score'] = f"{item['combined_score']:.4f}"
                doc.metadata['vector_score'] = f"{item['vector_score']:.4f}"
                doc.metadata['term_score'] = f"{item['term_score']:.2f}"
                
                filtered_docs.append(doc)
            
            print(f"[QDRANT TOOL] Retornando {len(filtered_docs)} documentos (de {len(results)} candidatos)")
            return filtered_docs
            
        except Exception as e:
            print(f"[QDRANT TOOL] ❌ ERROR: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    return tool_sync


try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None


class GeminiClient:


    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("Instala langchain-google-genai: pip install langchain-google-genai")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY no configurada en entorno")
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.0,
        )

    async def generate(self, prompt: str) -> str:
        """Genera una respuesta usando el LLM de Gemini."""
        loop = asyncio.get_event_loop()

        def sync_call():
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)

        return await loop.run_in_executor(None, sync_call)


class SimpleAgent:

    def __init__(self, llm: GeminiClient, memory: PostgresChatMemory, tool1, tool2, tool1_desc: str = "", tool2_desc: str = ""):
        self.llm = llm
        self.memory = memory
        self.tool1 = tool1
        self.tool2 = tool2
        self.tool1_desc = tool1_desc
        self.tool2_desc = tool2_desc
    
    async def expand_query(self, user_message: str) -> List[str]:
        """Expande la query del usuario con sinónimos y variaciones para mejorar búsqueda."""
        
        expansions = {
            'tarifa': ['tarifa', 'precio', 'costo', 'valor', 'monto'],
            'precio': ['precio', 'tarifa', 'costo', 'valor'],
            'auto': ['auto', 'vehículo', 'carro', 'automóvil'],
            'vehículo': ['vehículo', 'auto', 'carro', 'automóvil'],
            'daño': ['daño', 'avería', 'desperfecto', 'rotura'],
            'robo': ['robo', 'hurto', 'sustracción'],
            'accidente': ['accidente', 'colisión', 'choque', 'siniestro'],
            'póliza': ['póliza', 'seguro', 'cobertura', 'protección'],
            'cancelar': ['cancelar', 'anular', 'rescindir'],
            'reserva': ['reserva', 'reservación', 'booking'],
            'alquiler': ['alquiler', 'renta', 'arrendamiento'],
            'oficina': ['oficina', 'sucursal', 'punto de atención'],
        }
        
        query_lower = user_message.lower()
        expanded_terms = set([user_message])  #
        for term, synonyms in expansions.items():
            if term in query_lower:
                for syn in synonyms[:2]:  
                    if syn not in query_lower:
                        expanded_terms.add(f"{user_message} {syn}")
        
        result = list(expanded_terms)[:3]  
        if len(result) > 1:
            print(f"[QUERY EXPANSION] Query expandida a {len(result)} variaciones")
        return result
    
    async def classify_question(self, user_message: str) -> Dict[str, Any]:
        """Clasifica la pregunta del usuario para determinar filtros de metadatos."""
        
        legal_keywords = [
            'daño', 'daños', 'pérdida', 'pérdidas', 'robo', 'robos', 'responsabilidad',
            'seguro', 'cobertura', 'accidente', 'accidentes', 'legal', 'política',
            'políticas', 'término', 'términos', 'condición', 'condiciones', 'contrato',
            'cancelación', 'cancelar', 'modificar', 'modificación', 'requisito', 'requisitos',
            'licencia', 'edad', 'conductor', 'devolucion', 'devolución', 'reembolso',
            'penalidad', 'penalidades', 'multa', 'garantía', 'drop-off', 'vigencia',
            'seguridad', 'protección', 'proteger', 'medida', 'medidas', 'cumplir',
            'norma', 'normas', 'obligación', 'obligaciones', 'prohibición', 'prohibido',
            'mantenimiento', 'cuidado', 'uso', 'correcto', 'incorrecto'
        ]
        
        tarifa_keywords = [
            'tarifa', 'tarifas', 'precio', 'precios', 'costo', 'costos', 'pago',
            'cuánto cuesta', 'cuanto cuesta', 'disponibilidad', 'modelo', 'modelos',
            'categoría', 'categoria', 'oficina', 'oficinas', 'ubicación', 'ubicacion',
            'flota', 'vehículo', 'vehiculo', 'auto', 'autos', 'carro', 'carros',
            'económico', 'economico', 'medio', 'premium', 'jeep', 'minivan'
        ]
        
        message_lower = user_message.lower()
        
        legal_score = sum(1 for keyword in legal_keywords if keyword in message_lower)
        tarifa_score = sum(1 for keyword in tarifa_keywords if keyword in message_lower)
        
        print(f"[CLASIFICACIÓN] Legal: {legal_score}, Tarifas: {tarifa_score}")
        
        result = {
            'category': 'general',
            'kb1_filter': None,
            'kb2_filter': None,
            'prioritize': None
        }
        
        if legal_score > tarifa_score and legal_score > 0:
            result['category'] = 'legal'
            result['prioritize'] = 'kb1'
            print(f"[CLASIFICACIÓN] Pregunta clasificada como: LEGAL/POLÍTICAS")
        elif tarifa_score > legal_score and tarifa_score > 0:
            result['category'] = 'tarifas'
            result['prioritize'] = 'kb2'
            print(f"[CLASIFICACIÓN] Pregunta clasificada como: TARIFAS/OPERACIONES")
        else:
            print(f"[CLASIFICACIÓN] Pregunta clasificada como: GENERAL (buscar en ambas KB)")
        
        return result

    async def run(self, chat_id: str, user_message: str) -> str:
        await self.memory.add_message(chat_id, "user", user_message)

        recent = await self.memory.get_recent(chat_id, limit=8)

        classification = await self.classify_question(user_message)


        docs1 = []
        docs2 = []
        

        expanded_queries = await self.expand_query(user_message)

        k1 = 8 if classification['prioritize'] == 'kb1' else 5
        k2 = 5 if classification['prioritize'] == 'kb1' else 8
        
        score_threshold_kb1 = 0.0 
        score_threshold_kb2 = 0.0  
        
        print(f"\n[QDRANT] Buscando en KB-1 (Políticas y Legal) - k={k1}, threshold={score_threshold_kb1}")
        if self.tool1:
            try:
                main_query = expanded_queries[0]
                docs1 = self.tool1(main_query, k=k1, metadata_filter=classification['kb1_filter'], score_threshold=score_threshold_kb1)
                
                if len(docs1) < 2 and len(expanded_queries) > 1:
                    print(f"[QDRANT] Pocos resultados, intentando con query expandida...")
                    docs1_expanded = self.tool1(expanded_queries[1], k=k1, metadata_filter=classification['kb1_filter'], score_threshold=score_threshold_kb1)
                    existing_contents = {getattr(d, 'page_content', '') for d in docs1}
                    for doc in docs1_expanded:
                        if getattr(doc, 'page_content', '') not in existing_contents:
                            docs1.append(doc)
                
                print(f"[QDRANT] ✅ KB-1: Se encontraron {len(docs1)} documentos (ordenados por relevancia)")
                for i, doc in enumerate(docs1, 1):
                    content = getattr(doc, 'page_content', '') or str(doc)
                    metadata = getattr(doc, 'metadata', {})
                    combined_score = metadata.get('score', 'N/A')
                    vector_score = metadata.get('vector_score', 'N/A')
                    term_score = metadata.get('term_score', 'N/A')
                    
                    if content:
                        preview = content[:300] + "..." if len(content) > 300 else content
                        print(f"[QDRANT] KB-1 Doc #{i} [Combined:{combined_score} Vector:{vector_score} Terms:{term_score}]")
                        print(f"         {preview}\n")
                    else:
                        print(f"[QDRANT] KB-1 Doc #{i}:  CONTENIDO VACÍO\n")
            except Exception as e:
                print(f"[QDRANT]  Error en KB-1: {type(e).__name__}: {str(e)}")
                docs1 = []
        else:
            print(f"[QDRANT] KB-1: Herramienta no disponible")
        
        print(f"\n[QDRANT] Buscando en KB-2 (Operaciones y Tarifas) - k={k2}, threshold={score_threshold_kb2}")
        if self.tool2:
            try:
                main_query = expanded_queries[0]
                docs2 = self.tool2(main_query, k=k2, metadata_filter=classification['kb2_filter'], score_threshold=score_threshold_kb2)
                
                if len(docs2) < 2 and len(expanded_queries) > 1:
                    print(f"[QDRANT] Pocos resultados, intentando con query expandida...")
                    docs2_expanded = self.tool2(expanded_queries[1], k=k2, metadata_filter=classification['kb2_filter'], score_threshold=score_threshold_kb2)
                    existing_contents = {getattr(d, 'page_content', '') for d in docs2}
                    for doc in docs2_expanded:
                        if getattr(doc, 'page_content', '') not in existing_contents:
                            docs2.append(doc)
                
                print(f"[QDRANT] KB-2: Se encontraron {len(docs2)} documentos (ordenados por relevancia)")
                for i, doc in enumerate(docs2, 1):
                    content = getattr(doc, 'page_content', '') or str(doc)
                    metadata = getattr(doc, 'metadata', {})
                    combined_score = metadata.get('score', 'N/A')
                    vector_score = metadata.get('vector_score', 'N/A')
                    term_score = metadata.get('term_score', 'N/A')
                    
                    if content:
                        preview = content[:300] + "..." if len(content) > 300 else content
                        print(f"[QDRANT] KB-2 Doc #{i} [Combined:{combined_score} Vector:{vector_score} Terms:{term_score}]")
                        print(f"         {preview}\n")
                    else:
                        print(f"[QDRANT] KB-2 Doc #{i}:  CONTENIDO VACÍO\n")
            except Exception as e:
                print(f"[QDRANT]  Error en KB-2: {type(e).__name__}: {str(e)}")
                docs2 = []
        else:
            print(f"[QDRANT] KB-2: Herramienta no disponible")
        
        print(f"\n[QDRANT] Resumen: KB-1={len(docs1)} docs, KB-2={len(docs2)} docs\n")

        prompt_parts = [
            "1. ROL, IDENTIDAD Y OBJETIVO",
            "Identidad: Eres un Agente Virtual de Atención al Cliente altamente profesional para TRANSTUR, S.A. (marcas Cubacar, Havanautos y REX).",
            "",
            "Tono: Profesional y Absolutamente Preciso.",
            "",
            "Misión Principal: Responder todas las consultas de los clientes basándote EXCLUSIVAMENTE en la información de las bases de conocimiento.",
            "",
            "2. RESTRICCIONES DE CONOCIMIENTO (PROTOCOLO CRÍTICO)",
            "2.1. FUENTES DE CONOCIMIENTO",
            "Tu información proviene de dos bases de datos (Qdrant) que serán consultadas simultáneamente. Debes identificar la fuente de cada fragmento:",
            "",
            "KB-1: POLÍTICAS Y LEGAL (Términos, Condiciones de Renta, Políticas de Privacidad).",
            "",
            "KB-2: OPERACIONES Y TARIFAS (Datos Operacionales, Precios, Ubicaciones, Logística).",
            "",
            "2.2. REGLAS DE ORO",
            "PROHIBICIÓN ABSOLUTA: NUNCA debes inventar, adivinar, especular o utilizar conocimiento previo o externo a los fragmentos de texto recuperados de KB-1 y KB-2 PARA CONSULTAS SOBRE SERVICIOS, POLÍTICAS Y OPERACIONES.",
            "",
            "EXCEPCIÓN - USO DEL HISTORIAL: Para preguntas personales o de contexto conversacional (como '¿Sabes mi nombre?', '¿De qué estábamos hablando?', saludos, etc.), SÍ PUEDES y DEBES usar la información del HISTORIAL DE CONVERSACIÓN. El historial es tu memoria de la conversación actual con este cliente específico.",
            "",
            "FORMATO: Sintetiza la información clara y concisamente. Utiliza viñetas para respuestas que cubran múltiples puntos.",
            "",
            "2.3. MANEJO DE INFORMACIÓN PARCIAL",
            "REGLA DE ORO: Siempre intenta ayudar al cliente con la información disponible, aunque sea parcial o indirecta.",
            "",
            "- Si encuentras información RELACIONADA pero no exactamente lo que busca: Proporciona la información relacionada que tengas y explica cómo se relaciona con su consulta",
            "- Si la información es parcial: Comparte lo que sabes y ofrece contactar canales oficiales para más detalles",
            "- Solo si NO hay NADA relacionado (documentos completamente irrelevantes): Indica que no tienes esa información específica y proporciona contactos oficiales",
            "",
            "IMPORTANTE: Prioriza ser ÚTIL con información parcial o relacionada antes que decir que no tienes información. Si hay algo que pueda ayudar al cliente, compártelo.",
            "",
            "3. INSTRUCCIONES DE ACCIÓN Y PRIORIZACIÓN",
            "Al formular una respuesta, utiliza la siguiente jerarquía de acción y fuente:",
            "",
            "INFERENCIA INTELIGENTE: Si la pregunta es sobre un tema específico y no encuentras información directa, pero encuentras información relacionada (ej: pregunta sobre medidas de seguridad y encuentras info sobre responsabilidades, daños, seguros), usa esa información para dar una respuesta útil.",
            "",
            "CONTRATO Y VIGENCIA: Para preguntas sobre la duración del alquiler o el contrato, busca en KB-1."
            "",
            "MODIFICACIONES Y CANCELACIONES: Para cambios de reserva, busca en KB-1 e instruye al cliente a contactar a cubacar@transtur.cu (desde el correo de registro).",
            "",
            "TARIFAS Y DISPONIBILIDAD (Datos Variables): Si la consulta es sobre precios, tarifas, disponibilidad de modelos, o ubicaciones específicas de oficinas, prioriza la información de KB-2: OPERACIONES.",
            "",
            "GARANTÍA DEL VEHÍCULO: Si el cliente pregunta sobre modelos o marcas, busca en KB-1 y aclara que solo se garantiza la categoría del auto, no el modelo específico.",
            "",
            "PENALIDADES: Si se menciona la entrega en otra oficina (Drop-Off), busca en KB-1 e informa el cargo de drop-off más una penalidad del 50%.",
            "",
            "CANALES DE CONTACTO: Para cualquier pregunta sobre cómo contactar a la empresa, prioriza KB-1 y proporciona:",
            "INSTRUCCIÓN CLAVE DE FORMATO:",
            "FORMATO DE RESPUESTA: Responde SIEMPRE en texto plano, claro y profesional. NO uses formato Markdown.",
            "- NO uses asteriscos (*) para negritas o cursivas",
            "- NO uses almohadillas (#) para títulos",
            "- NO uses guiones bajos (_) para formato",
            "- Usa texto simple con viñetas (guiones -) cuando sea necesario",
            "- Separa secciones con saltos de línea simples",
            "- Mantén un tono profesional y amigable sin formatos especiales",
            "Cuando respondas con información extraída de la base de datos (ej. tarifas, detalles de productos), debes REESTRUCTURAR y REFORMATAR el texto en formato simple y claro, eliminando cualquier carácter de marcado interno que pueda confundir al usuario.",
            "",
            "---",
            "",
            f"CONTEXTO DE KB-1 (POLÍTICAS Y LEGAL): {self.tool1_desc}",
        ]
        
        if docs1:
            for d in docs1:
                content = getattr(d, 'page_content', '') or str(d)
                if content:
                    prompt_parts.append(f"- {content}")
        else:
            prompt_parts.append("(Sin información relevante en KB-1)")
        
        prompt_parts.append("")
        prompt_parts.append(f"CONTEXTO DE KB-2 (OPERACIONES Y TARIFAS): {self.tool2_desc}")
        
        if docs2:
            for d in docs2:
                content = getattr(d, 'page_content', '') or str(d)
                if content:
                    prompt_parts.append(f"- {content}")
        else:
            prompt_parts.append("(Sin información relevante en KB-2)")
        
        prompt_parts.append("")
        prompt_parts.append("HISTORIAL DE CONVERSACIÓN:")
        
        if recent:
            for m in recent:
                role = "Cliente" if m['role'] == "user" else "Agente"
                prompt_parts.append(f"{role}: {m['content']}")
        else:
            prompt_parts.append("(Primera interacción)")
        
        prompt_parts.append("")
        prompt_parts.append(f"CONSULTA ACTUAL DEL CLIENTE:")
        prompt_parts.append(user_message)
        prompt_parts.append("")
        prompt_parts.append("TU RESPUESTA (siguiendo todas las instrucciones anteriores):")

        prompt = "\n".join(prompt_parts)

        reply = await self.llm.generate(prompt)

        await self.memory.add_message(chat_id, "agent", reply)

        return reply


_agent: Optional[SimpleAgent] = None
_memory: Optional[PostgresChatMemory] = None


async def bootstrap() -> None:
    global _agent, _memory

    print("[BOOTSTRAP] Inicializando conexión a PostgreSQL...")
    if not POSTGRES_CONNECTION_STRING:
        print("[BOOTSTRAP]  POSTGRES_CONNECTION_STRING no configurada")
        _memory = None
    else:
        try:
            _memory = PostgresChatMemory(POSTGRES_CONNECTION_STRING)
            await _memory.init()
            print("[BOOTSTRAP]  Memoria PostgreSQL inicializada correctamente")
        except Exception as e:
            print(f"[BOOTSTRAP]  Error inicializando PostgreSQL: {type(e).__name__}: {str(e)}")
            _memory = None

    llm = None
    if GEMINI_API_KEY:
        try:
            gemini_client = GeminiClient(GEMINI_API_KEY)
            llm = gemini_client
        except Exception as e:
            print(f"No se pudo inicializar Gemini client: {e}")

    q_client = init_qdrant_client()
    embeddings = None
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GEMINI_API_KEY
        )
    except Exception as e:
        print(f"No se pudieron inicializar embeddings de Gemini: {e}")
        embeddings = None

    tool1 = None
    tool2 = None
    tool1_desc = "Contiene: Términos y Condiciones de Renta, Requisitos del Conductor, Políticas de Cancelación, Garantía de Vehículo, Contrato y Vigencia, Penalidades por Drop-Off, Canales de Contacto Oficiales."
    tool2_desc = "Contiene: Tarifas de Alquiler, Precios por Categoría, Disponibilidad de Modelos, Ubicaciones de Oficinas, Datos Operacionales de Flota, Información Logística."
    
    if q_client and embeddings:
        tool1 = create_retrieval_tool_from_collection(QDRANT_COLLECTION_1, q_client, embeddings)
        tool2 = create_retrieval_tool_from_collection(QDRANT_COLLECTION_2, q_client, embeddings)

    if llm and _memory:
        _agent = SimpleAgent(llm=llm, memory=_memory, tool1=tool1, tool2=tool2, tool1_desc=tool1_desc, tool2_desc=tool2_desc)
    else:
        print("AVISO: Agente no inicializado completamente. Revisa GEMINI_API_KEY y POSTGRES_CONNECTION_STRING.")


@app.on_event("startup")
async def on_startup():
    await bootstrap()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint para recibir mensajes desde el frontend.
    
    Args:
        request: ChatRequest con chat_id y mensaje del usuario
        
    Returns:
        ChatResponse con la respuesta del agente
    """
    print(f"[CHAT] Mensaje recibido de {request.chat_id}: {request.message}")
    
    try:
        if _agent is None:
            raise HTTPException(
                status_code=503, 
                detail="El servicio no está disponible. El agente no está inicializado."
            )
        
        reply = await _agent.run(request.chat_id, request.message)
        
        print(f"[CHAT] Respuesta generada para {request.chat_id}: {reply[:100]}...")
        
        return ChatResponse(chat_id=request.chat_id, response=reply)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[CHAT] ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error procesando mensaje: {str(e)}"
        )


@app.get("/chat/{chat_id}/history")
async def get_chat_history(chat_id: str, limit: int = 10):
    """
    Endpoint para obtener el historial de conversación de un chat.
    
    Args:
        chat_id: ID del chat
        limit: Número máximo de mensajes a recuperar (default: 10)
        
    Returns:
        Lista de mensajes del historial
    """
    print(f"[HISTORY] Solicitando historial de {chat_id} (limit={limit})")
    
    try:
        if _memory is None:
            raise HTTPException(
                status_code=503, 
                detail="El servicio de memoria no está disponible."
            )
        
        history = await _memory.get_recent(chat_id, limit=limit)
        
        print(f"[HISTORY] Recuperados {len(history)} mensajes para {chat_id}")
        
        return {"chat_id": chat_id, "history": history}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[HISTORY] ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error obteniendo historial: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """
    Endpoint para verificar el estado del servicio.
    
    Returns:
        Estado del servicio y sus componentes
    """
    return {
        "status": "ok",
        "agent_ready": _agent is not None,
        "memory_ready": _memory is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=5678, reload=True)