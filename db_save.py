from datetime import datetime
from db_init import get_db_connections, DB_TIMEZONE 

def save_conversation(record, question, course): # Changed 'questions' to 'question'
    timestamp = datetime.now(DB_TIMEZONE)

    conn = get_db_connections() # Pluralized to match your working db file
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations ( -- Fixed table name to plural
                    question, answer, course, model, instructions, prompt, prompt_tokens,
                    completion_tokens, total_tokens, response_time, cost, timestamp
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING id
                """,
                (
                    question, # Matches the updated parameter name above
                    record.answer,
                    course,
                    record.model,
                    record.instructions,
                    record.prompt,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.response_time,
                    record.cost,
                    timestamp,
                ),
            )
            conversation_id = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return conversation_id