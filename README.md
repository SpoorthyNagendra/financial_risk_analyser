# financial_risk_analyser
An intelligent financial analysis tool that leverages GPT-4o-mini and LangChain to provide AI-powered insights from SEC 10-K filings for any publicly traded company.

##  Overview
This application helps investors, analysts, and researchers quickly extract and understand key information from complex SEC 10-K filings through natural language queries. Built with production-grade ML monitoring and cost tracking.

## Key Features
Intelligent Document Analysis**: Semantic search across 10,000+ publicly traded companies' SEC filings
-  Conversational Interface**: Ask questions in natural language and get AI-powered answers
-  Real-time Market Data**: View current stock prices, market cap, and P/E ratios alongside risk analysis
-  Usage Analytics**: Track API costs, response times, token usage, and query patterns with MLflow
-  Conversation Export**: Download your analysis sessions for future reference
-  Performance Optimization**: Smart caching reduces analysis time from 60s to <2s for repeated queries
-  Production Monitoring**: Comprehensive error handling, logging, and retry logic

## Architecture
<pre>
```text
           ┌────────────┐
           │   User     │
           │ (Streamlit)│
           └─────┬──────┘
                 │
                 ▼
         ┌───────────────────┐        ┌───────────────┐
         │  Query Handler    ├───────▶│    OpenAI     │
         │   (LangChain)     │        │ (GPT-4o-mini) │
         └────────┬──────────┘        └───────────────┘
                  │
                  ▼
         ┌────────────────────┐        ┌──────────────────────┐
         │  Vector Database   ◄────────┤     Embeddings       │
         │     (FAISS)        │        │  (MiniLM-L6-v2 HF)   │
         └────────┬───────────┘        └──────────────────────┘
                  │
                  ▼
         ┌────────────────────┐        ┌────────────────────┐
         │  SEC EDGAR API     │        │      MLflow        │
         │   (10-K Data)      │        │     (Tracking)     │
         └────────────────────┘        └────────────────────┘
```
</pre>

## Getting Started
### Pre-requisites
- Python 3.8+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- 2GB+ free disk space (for caching and embeddings)

### Installation
1. ** Clone the Repository **
```bash
   git clone https://github.com/yourusername/financial-risk-analyzer.git
   cd financial-risk-analyzer
```
2. ** Create Virtual Environment **
```bash
   python -m venv venv
   source venv/bin/activate  # On Mac
   source venv\Scripts\activate #On Windows
```
3. ** Install Dependencies **
```bash
   pip install -r requirements.txt
```
4. ** Set Up Environment Variables **
```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
```
5. ** Run the application **
```bash
   streamlit run app.py
```
6. ** View mlflow tracking (optional) **
```bash
   mlflow ui
   # Open http://localhost:5000
```

## Usage
### Basic Usage
1. Enter a ticker(e.g. AAPL for Apple, MSFT for Microsoft, TSLA for Tesla etc..)
2. Press Enter should display found if the company found or INVALID if the ticker is invalid.
3. Select analyze 10-K fillings if company is found(takes ~60sec to 90sec first time, then gets cached once the data is loaded)
4. Ask questions related to the same ...Some examples of questions are :
   - What are main Business risks?
   - How does company generate revenue?
   - What regulatory Challenges does the company face?
   - What are top 3 cybersecurity risks?
   - Who are the main competitors of the company?
   - What regulatory investigations is the company facing?
5.View analytics in the analytics dashboard tab to track cost and usage.

## Analytics Dashboard
The built-in analytics dashboard provides insight into - 
- Cost Tracking : To monitor API expense for each query .
- Performance Metrics:To track response time and identify bottlenecks.
- Token Usage: Analyze input/output token consumption.

## Technical Stack
 | Component       | Technology|
-|  LLM            | OpenAI GPT 4o mini|
-| Framework       | Langchain |
-| Vector Database | FAISS |
-| Embeddings      | sentence-transformers/all-MiniLM-L6-v2|
-| FrontEnd        | streamlit|
-| Monitoring      | MLFlow |
-| Market Data     | Yahoo Finance(yfinance) |
-| Data Source     | SEC EDGAR API |

## Cost Estimates
Based on GPT-4o-mini pricing ($0.150/1M input tokens, $0.600/1M output tokens):

- Average query cost: $0.0001-0.0003 (~500-800 total tokens)
- 100 queries: ~$0.02-0.03
- 1000 queries: ~$0.20-0.30

Actual costs tracked in real-time via MLflow dashboard.

## Project Structure
```
financial-risk-analyzer/
├── app.py                 # Streamlit UI
├── main.py               # Core logic (analysis, queries)
├── tracking.py           # MLflow tracking decorator
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── .gitignore           # Git ignore rules
├── README.md            # This file
└── screenshots/         # Demo screenshots
```
## Security and Best Practices
- API keys stored in `.env` (never committed to Git)
- Comprehensive error handling with retry logic
- Rate limiting awareness for SEC API
- Input validation for ticker symbols
- Secure caching with data integrity checks
- Logging for debugging and monitoring

## Use Cases
- Investors: Quick risk assessment before making investment decisions
- Financial Analysts: Compare risks across multiple companies
- Researchers: Extract specific information from lengthy filings
- Compliance Teams: Monitor regulatory risk disclosures
- Due Diligence: Rapid company research for M&A activities

## Known Limitations
- Analysis limited to most recent 10-K filing
- Does not provide investment advice or recommendations
- Requires OpenAI API key (not free, but very affordable)
- First-time analysis of a company takes 30-60 seconds
- Market data delayed by ~15 minutes (Yahoo Finance limitation)

## Future Enhancements
- Support for 10-Q quarterly reports
- Multi-year trend analysis
- Automated risk scoring system
- Email alerts for new filings
- Company comparison dashboard
- Fine-tuned model for financial domain
- Integration with additional data sources

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- SEC EDGAR for providing free access to financial data
- OpenAI for GPT-4o-mini API
- LangChain for the RAG framework
- Streamlit for the intuitive UI framework

## Contact
- Email: spoorthynagendra@gmail.com
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/spoorthy-nagendra-ds)

**Note**: This tool is for informational purposes only and does not constitute financial advice. Always consult with a qualified financial advisor before making investment decisions.
