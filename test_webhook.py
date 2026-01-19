"""
test_webhook.py

Script de prueba para el webhook de WhatsApp.
Simula un mensaje de WAHA y verifica la respuesta del servidor.
"""

import httpx
import asyncio
import json

# Configuración
BASE_URL = "http://localhost:8000"

# Payload de ejemplo simulando un mensaje de WAHA
test_payload = {
    "event": "message",
    "message": {
        "from": "+1234567890",
        "body": "Hola, necesito información sobre las tarifas de alquiler de autos",
        "id": "test-message-001",
        "timestamp": 1700000000
    }
}


async def test_webhook():
    """Prueba el endpoint /webhook con un mensaje de ejemplo."""
    
    print("🧪 Iniciando prueba del webhook...")
    print(f"📍 URL: {BASE_URL}/webhook")
    print(f"📦 Payload:\n{json.dumps(test_payload, indent=2)}\n")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{BASE_URL}/webhook",
                json=test_payload
            )
            
            print(f"✅ Status Code: {response.status_code}")
            print(f"📄 Response:\n{json.dumps(response.json(), indent=2)}")
            
            if response.status_code == 200:
                print("\n✨ ¡Prueba exitosa!")
            else:
                print(f"\n⚠️  Respuesta inesperada: {response.status_code}")
                
    except httpx.ConnectError:
        print("❌ Error: No se pudo conectar al servidor.")
        print("   Asegúrate de que el servidor esté corriendo en http://localhost:8000")
        print("   Ejecuta: python main.py")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")


async def test_health():
    """Verifica que el servidor esté activo."""
    
    print("🏥 Verificando estado del servidor...\n")
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BASE_URL}/docs")
            
            if response.status_code == 200:
                print("✅ Servidor activo y respondiendo")
                print(f"📖 Documentación disponible en: {BASE_URL}/docs\n")
                return True
            else:
                print(f"⚠️  Servidor respondió con código: {response.status_code}\n")
                return False
                
    except httpx.ConnectError:
        print("❌ Servidor no disponible")
        print("   Ejecuta: python main.py\n")
        return False
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False


async def main():
    """Ejecuta todas las pruebas."""
    
    print("=" * 60)
    print("🤖 PRUEBA DE WEBHOOK - CHATBOT WHATSAPP")
    print("=" * 60 + "\n")
    
    # Verificar que el servidor esté activo
    if await test_health():
        # Probar el webhook
        await test_webhook()
    else:
        print("⏭️  Saltando prueba del webhook (servidor no disponible)")
    
    print("\n" + "=" * 60)
    print("✅ Pruebas completadas")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
