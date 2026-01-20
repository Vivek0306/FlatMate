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

    # Display answer
    print(f"\n✅ Answer:\n{result['answer']}")
    
    # Display sources
    # if 'context' in result and result['context']:
    #     print(f"\n📚 Sources Used:")
    #     sources = set()
    #     for doc in result['context']:
    #         source = doc.metadata.get('source', 'Unknown')
    #         page = doc.metadata.get('page', None)
    #         if page is not None:
    #             sources.add(f"  • {source} (Page {page + 1})")
    #         else:
    #             sources.add(f"  • {source}")
        
    #     for source in sorted(sources):
    #         print(source)
    
    print("-" * 100)