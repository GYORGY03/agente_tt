# 🤖 Chatbot WhatsApp con FastAPI + LangChain + Gemini

Sistema completo de chatbot para WhatsApp que utiliza:
- **FastAPI** para el servidor webhook
- **WAHA** para la integración con WhatsApp
- **LangChain 1.0** para el agente de IA
- **Google Gemini** como modelo de lenguaje
- **Qdrant** para RAG (Retrieval-Augmented Generation) con dos bases de conocimiento
- **PostgreSQL** para memoria persistente de conversaciones

## 📋 Requisitos Previos

- Python 3.13.9 o superior
- PostgreSQL instalado y corriendo
- Qdrant instalado y corriendo (o acceso a instancia cloud)
- Instancia de WAHA configurada y activa
- API Key de Google Gemini

## 🚀 Instalación

### 1. Clonar o preparar el proyecto

```powershell
cd C:\Users\jorge\Documents\Tt
```

### 2. Crear y activar entorno virtual

```powershell
python -m venv .env
.\.env\Scripts\activate
```

### 3. Instalar dependencias

```powershell
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Copia el archivo `.env.example` a `.env` y completa con tus credenciales:

```powershell
copy .env.example .env
```

Edita `.env` con tus valores reales:

```env
GEMINI_API_KEY=tu_api_key_de_gemini
WAHA_API_URL=http://localhost:3000
POSTGRES_CONNECTION_STRING=postgresql://user:password@localhost:5432/chatbot_db
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=tu_api_key_si_aplica
QDRANT_COLLECTION_1=tarifas_autos
QDRANT_COLLECTION_2=contexto_general
```

### 5. Preparar la base de datos PostgreSQL

Crea la base de datos (la tabla se crea automáticamente al iniciar):

```sql
CREATE DATABASE chatbot_db;
```

### 6. Preparar las colecciones de Qdrant

Asegúrate de que tus dos colecciones estén creadas y pobladas con embeddings de Google Gemini.

## 🎯 Uso

### Iniciar el servidor

```powershell
python main.py
```

O usando uvicorn directamente:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

El servidor estará disponible en `http://localhost:8000`

### Configurar WAHA

En tu instancia de WAHA, configura un webhook que apunte a:

```
http://tu-servidor:8000/webhook
```

### Probar el endpoint

Puedes probar el webhook enviando un POST:

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/webhook" -ContentType "application/json" -Body '{
  "event": "message",
  "message": {
    "from": "+1234567890",
    "body": "Hola, necesito información sobre alquiler de autos"
  }
}'
```

## 📁 Estructura del Proyecto

```
Tt/
├── main.py              # Aplicación principal con FastAPI y LangChain
├── .env                 # Variables de entorno (no incluir en git)
├── .env.example         # Plantilla de variables de entorno
├── requirements.txt     # Dependencias del proyecto
└── README.md           # Este archivo
```

## 🔧 Componentes Principales

### 1. **Webhook FastAPI** (`/webhook`)
Recibe eventos de WAHA cuando llega un mensaje de WhatsApp.

### 2. **Agente LangChain** (`SimpleAgent`)
- Mantiene contexto de conversación por usuario
- Consulta dos bases de conocimiento en Qdrant
- Genera respuestas usando Gemini

### 3. **Memoria PostgreSQL** (`PostgresChatMemory`)
- Almacena historial de conversaciones por `chat_id`
- Tabla: `chat_messages` (creada automáticamente)

### 4. **Tools RAG** (Qdrant)
- Tool 1: Primera colección (ej: tarifas de autos)
- Tool 2: Segunda colección (ej: contexto general)

### 5. **Envío a WhatsApp** (`send_whatsapp_message`)
- Envía respuestas del agente vía API de WAHA

## 🔍 Ejemplo de Payload de WAHA

```json
{
  "event": "message",
  "message": {
    "from": "+1234567890",
    "body": "Hola, quiero información sobre alquiler de autos",
    "id": "abcd-1234",
    "timestamp": 1690000000
  }
}
```

## 🛠️ Personalización

### Cambiar el modelo de Gemini

Edita en `GeminiClient.generate()`:

```python
model: str = "gemini-2.5-flash"  # Cambia a otro modelo disponible
```

### Ajustar límite de historial

En `SimpleAgent.run()`:

```python
recent = await self.memory.get_recent(chat_id, limit=8)  # Cambia el límite
```

### Modificar el prompt del agente

Edita la construcción del prompt en `SimpleAgent.run()`:

```python
prompt_parts = [
    "Eres un asistente conversacional que responde de forma clara y breve.",
    # Personaliza según tus necesidades
]
```

## 📝 Notas Importantes

1. **Adaptaciones necesarias**: El código usa wrappers simplificados para Gemini. Ajusta según la versión exacta de `google-genai` o `langchain-google-genai` que uses.

2. **Seguridad**: Nunca commitees el archivo `.env` a git. Añádelo a `.gitignore`.

3. **Producción**: Para producción, considera:
   - Usar un servidor ASGI como gunicorn con workers uvicorn
   - Implementar rate limiting
   - Añadir logging apropiado
   - Manejar errores de forma más robusta
   - Usar variables de entorno del sistema en lugar de archivos .env

4. **LangChain avanzado**: Este ejemplo usa un agente simplificado. Para mayor robustez, considera usar `AgentExecutor` de LangChain con herramientas definidas usando `@tool` decorators.

## 🐛 Troubleshooting

### Error: "GEMINI_API_KEY no configurada"
Verifica que el archivo `.env` exista y contenga tu API key.

### Error de conexión a PostgreSQL
Verifica que PostgreSQL esté corriendo y que la cadena de conexión sea correcta.

### Error de conexión a Qdrant
Asegúrate de que Qdrant esté accesible en la URL configurada.

### WAHA no envía webhooks
Verifica que la URL del webhook esté correctamente configurada en WAHA y que tu servidor sea accesible desde la instancia de WAHA.

## 📚 Referencias

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [LangChain Docs](https://python.langchain.com/)
- [WAHA Docs](https://waha.devlike.pro/)
- [Google Gemini API](https://ai.google.dev/)
- [Qdrant Docs](https://qdrant.tech/documentation/)

## 📄 Licencia

Este proyecto es un ejemplo educativo. Úsalo como base para tu proyecto.
