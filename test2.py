from llm_reponse import generate_reply

# Simulate a real field email from a Sopan team
query = """
Subject: Element Damage Observed in BHA Post Trip-Out

We observed significant scraping and deep scoring on the outer cage and setting cone of the BHA after pulling out of hole.
Formation: Midale
BHA Type: SFC
Packer Type: Mongoose
Observed Sleeve Shift Pressure: ~3050 psi

Please advise on possible causes and corrective actions.
"""

# Call your RAG bot
response = generate_reply(query)

# Output result
print("\n--- Final RAG Bot Response ---\n")
print(response)