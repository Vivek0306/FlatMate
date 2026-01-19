# app.py

import threading
from rag_app import create_rag_chain
from spinner import spinner
from config import APP_NAME, APP_TAGLINE

print(f"Loading {APP_NAME} - {APP_TAGLINE}")
rag_chain = create_rag_chain()
print("✓ Ready!")

print("\n" + "=" * 60)
print(f"{APP_NAME.upper()} - {APP_TAGLINE.upper()}")
print("=" * 60)

while True:
    question = input("\n❓ Ask a question (or type 'exit'): ")
    if question.lower() == "exit":
        print("Goodbye 👋")
        break

    stop_event = threading.Event()
    t = threading.Thread(target=spinner, args=(stop_event,))
    t.start()

    try:
        result = rag_chain.invoke({"input": question})
    finally:
        stop_event.set()
        t.join()

    print(f"\n✅ Answer:\n{result['answer']}")
    print("-" * 100)
