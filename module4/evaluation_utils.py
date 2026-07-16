import json
import time

from tqdm.auto import tqdm
from module4.rag_helper import RAGBase


def calc_price(usage):
    # Groq pricing (approximate, check console.groq.com for current rates)
    input_price_per_million = 0.60
    output_price_per_million = 3.00
    ...

    input_cost = (usage.prompt_tokens / 1_000_000) * input_price_per_million
    output_cost = (usage.completion_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }



def calc_total_price(usages):
    total_cost = 0.0

    for usage in usages:
        cost = calc_price(usage)
        total_cost = total_cost + cost["total_cost"]

    return total_cost


def llm_structured(client, instructions, user_prompt, output_type, model="qwen/qwen3.6-27b"):
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        max_completion_tokens=2000,
        extra_body={"reasoning_format": "hidden", "reasoning_effort": "none"}
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    result = output_type(**data)

    return result, response.usage


import time  # Make sure this import is at the top of evaluation_utils.py

def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model="qwen/qwen3.6-27b",
    max_retries=3,
):
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client,
                instructions,
                user_prompt,
                output_type,
                model=model,
            )
        except Exception as e:
            # If we hit a rate limit, wait longer on each retry attempt
            if attempt < max_retries - 1:
                sleep_time = (attempt + 1) * 15  # Waits 15s on 1st fail, 30s on 2nd
                print(f"Rate limit or API error encountered. Retrying in {sleep_time} seconds... (Error: {e})")
                time.sleep(sleep_time)
            else:
                # If we've run out of retries, raise the exception
                raise e

class RAGWithUsage(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        boost_dict = {"question": 1.0, "answer": 2.0, "section": 0.1}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def llm(self, prompt):
        input_messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=input_messages
        )

        self.last_usage = response.usage
        self.usages.append(response.usage)

        return response.choices[0].message.content

    def total_cost(self):
        return calc_total_price(self.usages)


def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)

    return results