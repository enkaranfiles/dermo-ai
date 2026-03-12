"""
CLI entry point for the Dermatoloji Chatbot PoC.

Usage:
    python main.py                  # interactive chat
    python main.py --image path.jpg # chat with skin image analysis
    python main.py --serve          # start FastAPI server
"""

import argparse
import sys
import os

from dotenv import load_dotenv
load_dotenv()  # loads ANTHROPIC_API_KEY from .env

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(__file__))


def run_cli(image_path: str | None = None) -> None:
    from chat.conversation_manager import ConversationManager
    from pipeline.diagnosis_pipeline import process_user_message

    manager = ConversationManager()

    # Print greeting
    greeting = manager.build_greeting()
    manager.add_assistant_message(greeting)
    print("\n🤖 Asistan:", greeting)
    print("\n" + "─" * 60)

    while True:
        try:
            user_input = input("\n👤 Siz: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGörüşmek üzere! Sağlıklı günler dileriz.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("çıkış", "exit", "quit", "q"):
            print("\nGörüşmek üzere! Sağlıklı günler dileriz.")
            break

        # Only use image on first user turn
        img = image_path if manager.turn_count == 0 else None

        print("\n🤖 Asistan: ", end="", flush=True)
        reply = process_user_message(
            manager=manager,
            user_input=user_input,
            image_path=img,
        )
        print(reply)
        print("\n" + "─" * 60)

        if manager.diagnosis_done:
            cont = input("\nDevam etmek ister misiniz? (e/h): ").strip().lower()
            if cont not in ("e", "evet", "yes", "y"):
                print("\nGörüşmek üzere! Sağlıklı günler dileriz.")
                break
            manager.diagnosis_done = False  # Allow follow-up turns


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    print(f"\n🚀 Dermatoloji Chatbot API başlatılıyor: http://{host}:{port}")
    print("📚 API Docs: http://localhost:{port}/docs\n")
    uvicorn.run(
        "api.fastapi_app:app",
        host=host,
        port=port,
        reload=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dermatoloji Chatbot PoC — Türkçe deri hastalığı bilgi asistanı"
    )
    parser.add_argument(
        "--image",
        metavar="PATH",
        help="Analiz edilecek deri görüntüsünün yolu (opsiyonel)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="FastAPI sunucusunu başlat",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Sunucu adresi (varsayılan: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Sunucu portu (varsayılan: 8000)")

    args = parser.parse_args()

    if args.serve:
        run_server(host=args.host, port=args.port)
    else:
        print("=" * 60)
        print("  Dermatoloji Bilgi Asistanı  |  PoC v0.1")
        print("  NOT: Bu sistem tıbbi teşhis aracı DEĞİLDİR.")
        print("=" * 60)
        run_cli(image_path=args.image)


if __name__ == "__main__":
    main()
