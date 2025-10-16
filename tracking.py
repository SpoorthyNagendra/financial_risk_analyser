import mlflow
import time
from functools import wraps
import tiktoken

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("financial-risk-analyzer")

# Initialize tokenizer for cost calculation
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

def estimate_cost(prompt_tokens, completion_tokens):
    """
    GPT-4o-mini pricing:
    - Input: $0.150 per 1M tokens
    - Output: $0.600 per 1M tokens
    """
    input_cost = (prompt_tokens / 1_000_000) * 0.150
    output_cost = (completion_tokens / 1_000_000) * 0.600
    return input_cost + output_cost

def track_query(func):
    @wraps(func)
    def wrapper(vectorstore, company_name, question, stock_info=None):
        with mlflow.start_run():
            start_time = time.time()
            
            # Log parameters
            mlflow.log_param("company", company_name)
            mlflow.log_param("question_length", len(question))
            mlflow.log_param("model", "gpt-4o-mini")
            mlflow.log_param("model_version", "2024-07-18")
            
            # Count input tokens
            try:
                # Reconstruct the prompt to count tokens
                relevant_docs = vectorstore.similarity_search(question, k=3)
                context = "\n\n".join([doc.page_content[:800] for doc in relevant_docs])
                
                # Add stock context if applicable
                price_keywords = ['current', 'stock price', 'value now', 'price now', 'worth', 'trading']
                if stock_info and any(keyword in question.lower() for keyword in price_keywords):
                    stock_context = f"""
Current Stock Data:
- Current Price: ${stock_info['price']}
- Day High: ${stock_info['day_high']}
- Day Low: ${stock_info['day_low']}
- Market Cap: ${stock_info['market_cap']}
- P/E Ratio: {stock_info['pe_ratio']}
"""
                    context = stock_context + "\n\n" + context
                
                full_prompt = f"""Based on {company_name}'s 10-K filing and available data, answer this question concisely:

Context: {context}

Question: {question}"""
                
                prompt_tokens = len(encoding.encode(full_prompt))
                mlflow.log_metric("prompt_tokens", prompt_tokens)
                
            except Exception as e:
                mlflow.log_param("token_count_error", str(e))
                prompt_tokens = 0
            
            # Execute query
            try:
                result = func(vectorstore, company_name, question, stock_info)
                
                # Count output tokens
                completion_tokens = len(encoding.encode(result))
                mlflow.log_metric("completion_tokens", completion_tokens)
                
                # Calculate cost
                total_cost = estimate_cost(prompt_tokens, completion_tokens)
                mlflow.log_metric("estimated_cost_usd", total_cost)
                
                # Log success
                mlflow.log_param("status", "success")
                
            except Exception as e:
                # Log error
                mlflow.log_param("status", "error")
                mlflow.log_param("error_message", str(e))
                mlflow.log_param("error_type", type(e).__name__)
                raise
            
            # Log metrics
            latency = time.time() - start_time
            mlflow.log_metric("latency_seconds", latency)
            mlflow.log_metric("response_length", len(result))
            
            return result
    return wrapper