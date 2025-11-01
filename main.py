from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="tinyllama")
template = """You are an expert IoT tutor. The following is IoT study material and a student's question about it. 
Please provide a clear, detailed, and educational response.

 Study Material:
{study_material}

Student Question: {question}

Please:
1. Identify the relevant section(s) from the study material
2. Provide a comprehensive but clear explanation
3. Include examples where appropriate
4. Break down complex concepts into simpler terms
5. If the question relates to technical specifications or protocols, explain their practical applications
6. Reference specific parts of the study material to support your answer
7. If the topic involves multiple concepts, show how they interconnect
8. Suggest follow-up questions the student might want to explore

Remember to:
- Be clear and concise
- Use simple language while maintaining technical accuracy
- Provide real-world examples when possible
- Break down complex topics into manageable parts
- Encourage critical thinking

Response:"""

prompt = ChatPromptTemplate.from_template(template)
chain= prompt | model

while True:
    print("\n\n-------------------------------------")
    question=input("Ask your questions (q to quit): ")
    print("\n\n")
    if question=="q":
        break

    docs = retriever.invoke(question)
    
    study_material = "\n\n".join([doc.page_content for doc in docs])
    result = chain.invoke({"study_material": study_material, "question": question})
    print(result)