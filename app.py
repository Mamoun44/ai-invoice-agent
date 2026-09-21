import os
import psycopg2
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load configuration from .env
load_dotenv()

# 2. Database Connection Helper
def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        database=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        port=os.environ.get("DB_PORT")
    )

# 3. Define the Python Functions (Tools)
def get_invoice_amount(company_name: str, invoice_id: str) -> float:
    print(f"  [DB Query] Looking up invoice {invoice_id} for {company_name}...")
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT amount FROM invoices WHERE company_name = %s AND invoice_id = %s;",
            (company_name, invoice_id)
        )
        result = cur.fetchone()
        return float(result[0]) if result else 0.0
    finally:
        cur.close()
        conn.close()

def get_total_invoices_amount(company_name: str) -> float:
    """Returns the total amount across all invoices for the specified company."""
    print(f"  [DB Query] Calculating total for {company_name}...")
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT SUM(amount) FROM invoices WHERE company_name = %s;",
            (company_name,)
        )
        result = cur.fetchone()
        return float(result[0]) if result and result[0] is not None else 0.0
    finally:
        cur.close()
        conn.close()

AVAILABLE_TOOLS = {
    "get_invoice_amount": get_invoice_amount,
    "get_total_invoices_amount": get_total_invoices_amount
}

# 4. Agent Logic
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def ask_invoice_agent(current_company: str, question: str):
    print(f"\nUser [{current_company}]: {question}")
    
    system_prompt = (
        f"You are a helpful accounting assistant. "
        f"The user is currently logged into the company account: '{current_company}'. "
        f"Whenever you call a tool, you MUST use '{current_company}' as the company_name argument."
    )
    
    chat = client.chats.create(
        model="gemini-3.6-flash",
        config=types.GenerateContentConfig(
            tools=[get_invoice_amount, get_total_invoices_amount],
            system_instruction=system_prompt,
            temperature=0.0
        )
    )
    
    response = chat.send_message(question)
    
    # 5. The JSON Interception Loop
    if response.function_calls:
        for tool_call in response.function_calls:
            function_name = tool_call.name
            arguments = tool_call.args
            
            # Execute the matching database query
            function_to_call = AVAILABLE_TOOLS[function_name]
            tool_result = function_to_call(**arguments)
            print(f"  [DB Result] Returned {tool_result} to the AI")
            
            # Send the raw data back to the LLM
            response = chat.send_message(
                types.Part.from_function_response(
                    name=function_name,
                    response={"result": tool_result}
                )
            )
            
    print(f"Agent: {response.text}")


# 6. Run the Application
if __name__ == "__main__":
    print("=== Welcome to the AI Invoice Agent ===")
    
    # Let the user pick which company they are simulating
    current_company = input("Enter your company name (e.g., TechNova, Stark Industries, Globex, Acme Corp): ").strip()
    
    print(f"\nLogged in as [{current_company}]. Type your questions below. Type 'exit' to quit.\n")
    
    while True:
        try:
            user_question = input(f"[{current_company}] You: ").strip()
            
            if user_question.lower() == 'exit':
                print("Goodbye!")
                break
            
            if not user_question:
                continue
                
            ask_invoice_agent(current_company, user_question)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error]: {e}\n")