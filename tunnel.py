#!/usr/bin/env python3
"""Tunnel local API to public URL using ngrok."""
import time
from pyngrok import ngrok

# Set ngrok auth token (get free from https://dashboard.ngrok.com/get-started/your-authtoken)
# Uncomment and paste your token below:
# ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN_HERE")

try:
    print("🚀 Starting public tunnel...")
    
    # Create tunnel to localhost:8000
    public_url = ngrok.connect(8000)
    print(f"\n✅ PUBLIC URL: {public_url}")
    print(f"\n📡 Endpoints:")
    print(f"   Health: {public_url}/health")
    print(f"   Chat:   {public_url}/chat")
    print(f"   UI:     {public_url}/ui")
    print(f"\n⏱️  Tunnel will stay active for 2 hours (free tier)")
    print(f"   Press CTRL+C to close tunnel\n")
    
    # Keep tunnel open
    ngrok_process = ngrok.get_ngrok_process()
    ngrok_process.proc.wait()
    
except KeyboardInterrupt:
    print("\n\n👋 Closing tunnel...")
    ngrok.kill()
    print("✅ Tunnel closed")
except Exception as e:
    print(f"❌ Error: {e}")
    ngrok.kill()
