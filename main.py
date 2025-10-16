from dotenv import load_dotenv
import os
import requests
import json
import pickle
import yfinance as yf
import logging
import time
from functools import wraps
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tracking import track_query

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def handle_errors(func):
    """Decorator for consistent error handling across functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.Timeout:
            logger.error(f"Timeout in {func.__name__}")
            return None, None, "Request timed out. Please try again."
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error in {func.__name__}")
            return None, None, "Unable to connect to server. Check your internet connection."
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error in {func.__name__}: {e.response.status_code}")
            if e.response.status_code == 429:
                return None, None, "Rate limit exceeded. Please wait a few minutes."
            elif e.response.status_code == 404:
                return None, None, "Resource not found. The filing may not exist."
            else:
                return None, None, f"Server error: {e.response.status_code}"
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            return None, None, f"An unexpected error occurred: {str(e)}"
    return wrapper

# Initialize OpenAI
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0.7
)

# Initialize text processing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load SEC ticker database
def load_sec_ticker_map():
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers={'User-Agent': 'Spoorthy Nagendra your.email@example.com'})
    data = json.loads(response.text)
    
    ticker_map = {}
    for entry in data.values():
        ticker = entry['ticker']
        cik = str(entry['cik_str']).zfill(10)
        name = entry['title']
        ticker_map[ticker.upper()] = {'cik': cik, 'name': name}
    
    return ticker_map

TICKER_MAP = load_sec_ticker_map()
print(f"Loaded {len(TICKER_MAP)} companies")

# Get latest 10-K filing
def get_latest_10k_accession(cik):
    """
    Retrieve the most recent 10-K filing accession number for a company.
    
    Args:
        cik (str): Central Index Key (10-digit company identifier)
        
    Returns:
        str: Accession number (e.g., '0000320193-23-000106'), None if not found
        
    Raises:
        requests.RequestException: If SEC API request fails
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {'User-Agent': 'Spoorthy Nagendra spoorthynagendra@gmail.com'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        filings = data['filings']['recent']
        for i, form in enumerate(filings['form']):
            if form == '10-K':
                accession = filings['accessionNumber'][i]
                logger.info(f"Found 10-K filing: {accession} for CIK {cik}")
                return accession
        
        logger.warning(f"No 10-K filing found for CIK {cik}")
        return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get filings for CIK {cik}: {str(e)}")
        raise

# Analyze any company
# Create cache directory
if not os.path.exists('cache'):
    os.makedirs('cache')

@handle_errors
def analyze_company(ticker):
    """
    Analyze a company's 10-K filing and create searchable vector store.
    
    This function downloads the latest 10-K filing from SEC EDGAR, processes
    it into chunks, creates embeddings, and stores them in a FAISS vector
    database for semantic search. Results are cached for faster subsequent access.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'TSLA')
        
    Returns:
        tuple: (vectorstore, company_name, error_message)
            - vectorstore (FAISS): Vector database with document embeddings, None if error
            - company_name (str): Full company name, None if error
            - error_message (str): Error description if failed, None if successful
    
    Examples:
        >>> vectorstore, name, error = analyze_company('AAPL')
        >>> if not error:
        ...     print(f"Successfully analyzed {name}")
        
    Notes:
        - First analysis takes 30-60 seconds (downloads and processes filing)
        - Subsequent analyses are instant (loaded from cache)
        - Cache stored in ./cache/ directory
    """
    ticker = ticker.upper()
    cache_file = f"cache/{ticker}_vectorstore.pkl"
    
    # Check cache first
    if os.path.exists(cache_file):
        logger.info(f"Loading cached data for {ticker}")
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Successfully loaded cached data for {ticker}")
            return data['vectorstore'], data['company_name'], None
        except Exception as e:
            logger.warning(f"Corrupted cache file for {ticker}, will reprocess: {str(e)}")
            try:
                os.remove(cache_file)
            except:
                pass
    
    # Validate ticker exists
    if ticker not in TICKER_MAP:
        logger.warning(f"Invalid ticker: {ticker}")
        return None, None, f"Ticker '{ticker}' not found in SEC database"
    
    company = TICKER_MAP[ticker]
    cik = company['cik']
    name = company['name']
    
    logger.info(f"Starting analysis for {name} ({ticker})")
    
    # Get latest 10-K accession number
    try:
        accession = get_latest_10k_accession(cik)
        if not accession:
            logger.error(f"No 10-K filing found for {ticker}")
            return None, None, f"No 10-K filing available for {name}"
    except Exception as e:
        logger.error(f"Failed to get 10-K accession for {ticker}: {str(e)}")
        return None, None, "Failed to retrieve filing information from SEC"
    
    # Download 10-K with retry logic
    logger.info(f"Downloading 10-K filing for {ticker}")
    max_retries = 3
    response = None
    
    for attempt in range(max_retries):
        try:
            accession_clean = accession.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{accession}.txt"
            headers = {'User-Agent': 'Spoorthy Nagendra spoorthynagendra@gmail.com'}
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            logger.info(f"Successfully downloaded 10-K for {ticker} (attempt {attempt + 1})")
            break
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1} for {ticker}")
            if attempt == max_retries - 1:
                return None, None, "Download timed out after multiple attempts. Please try again later."
            time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed on attempt {attempt + 1} for {ticker}: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    
    if not response:
        return None, None, "Failed to download filing after multiple attempts"
    
    # Validate response
    if len(response.text) < 1000:
        logger.error(f"Downloaded filing for {ticker} appears invalid (too short)")
        return None, None, "Downloaded filing appears incomplete. Please try again."
    
    # Process document into chunks
    logger.info(f"Processing document for {ticker}")
    try:
        chunks = text_splitter.split_text(response.text)
        
        if not chunks:
            logger.error(f"No chunks created for {ticker}")
            return None, None, "Failed to process document"
        
        logger.info(f"Created {len(chunks)} chunks for {ticker}")
        
    except Exception as e:
        logger.error(f"Failed to split text for {ticker}: {str(e)}")
        return None, None, "Failed to process document text"
    
    # Create vector store
    logger.info(f"Creating vector embeddings for {ticker}")
    try:
        vectorstore = FAISS.from_texts(chunks, embeddings)
        logger.info(f"Successfully created vector store for {ticker}")
        
    except Exception as e:
        logger.error(f"Failed to create embeddings for {ticker}: {str(e)}")
        return None, None, "Failed to create searchable database"
    
    # Cache the results
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({'vectorstore': vectorstore, 'company_name': name}, f)
        logger.info(f"Successfully cached data for {ticker}")
    except Exception as e:
        logger.warning(f"Failed to cache data for {ticker}: {str(e)}")
        # Don't fail if caching fails, just log it
    
    return vectorstore, name, None

# Query function
@track_query
def query_company(vectorstore, company_name, question, stock_info=None):
    """
    Query a company's 10-K filing using natural language and get AI-powered answers.
    
    This function performs semantic search on the vector database to find relevant
    sections of the 10-K filing, then uses GPT-4o-mini to generate a concise answer
    based on the retrieved context.
    
    Args:
        vectorstore (FAISS): Vector database containing company filing embeddings
        company_name (str): Full name of the company being queried
        question (str): Natural language question about the company
        stock_info (dict, optional): Current stock market data to include in context
        
    Returns:
        str: AI-generated answer based on 10-K filing content
        
    Examples:
        >>> answer = query_company(vs, "Apple Inc.", "What are the main risks?")
        >>> print(answer)
        
    Notes:
        - Queries are automatically tracked in MLflow with cost and latency metrics
        - Searches top 3 most relevant document chunks (k=3)
        - Includes stock data in context if question contains price-related keywords
        - Average response time: 2-5 seconds
    """
    relevant_docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content[:800] for doc in relevant_docs])
    
    # Add stock data to context if question is about current stock price or value
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
    
    prompt = f"""Based on {company_name}'s 10-K filing and available data, answer this question concisely:

Context: {context}

Question: {question}"""
    
    response = llm.invoke(prompt)
    return response.content


def get_stock_info(ticker):
    """
    Fetch current stock market data from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Stock information with keys: price, day_high, day_low, market_cap, pe_ratio
              None if data unavailable
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        stock_data = {
            'price': info.get('currentPrice', 'N/A'),
            'day_high': info.get('dayHigh', 'N/A'),
            'day_low': info.get('dayLow', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A')
        }
        
        logger.info(f"Successfully fetched stock info for {ticker}")
        return stock_data
        
    except Exception as e:
        logger.error(f"Failed to fetch stock info for {ticker}: {str(e)}")
        return None

# Test the system
if __name__ == "__main__":
    ticker = input("Enter company ticker: ")
    vectorstore, company_name, error = analyze_company(ticker)
    
    if error:
        print(error)
    else:
        question = input("Ask a question about the company: ")
        answer = query_company(vectorstore, company_name, question)
        print(f"\n{company_name} Analysis:\n{answer}")