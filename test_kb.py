from kb_store import search_kb, search_kb_raw

queries = [
    "Where is my order? It was supposed to arrive yesterday.",
    "I want to return the shoes I bought last week. How do I get a refund?",
    "I forgot my password and can't log in.",
    "The blender I received is broken. Can I get a replacement?",
    "You charged my card three times! Fix this now or I'm contacting my bank and a lawyer.",
    "Do you sell gift cards for a company of 500 employees with custom branding?",
]

for q in queries:
    print("\n" + "=" * 80)
    print("QUERY:", q)
    for r in search_kb_raw(q):
        print(f"  {r['score']:.3f}  {r['source']:<24} {r['text'][:70]!r}")

print("\n" + "=" * 80)
print("search_kb() output for the password question:\n")
print(search_kb(queries[2]))