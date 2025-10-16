import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from main import analyze_company, query_company, TICKER_MAP, get_stock_info

st.set_page_config(page_title="Financial Risk Analyzer", layout="wide")

st.title("LLM-Powered Financial Risk Analyzer")
st.markdown("Analyze SEC 10-K filings for any publicly traded company using AI")

# Initialize conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Sidebar - ALWAYS VISIBLE
with st.sidebar:
    st.header("Company Selection")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA)", value="").upper()

    if ticker and ticker in TICKER_MAP:
        st.success(f"✓ Found: {TICKER_MAP[ticker]['name']}")
    elif ticker:
        st.error("Ticker not found in SEC database")

    analyze_button = st.button(
        "Analyze 10-K Filing",
        type="primary",
        disabled=not ticker or ticker not in TICKER_MAP
    )

    if analyze_button:
        with st.spinner(f"Loading {TICKER_MAP[ticker]['name']}'s 10-K filing..."):
            vectorstore, company_name, error = analyze_company(ticker)
            if error:
                st.error(error)
            else:
                st.session_state.vectorstore = vectorstore
                st.session_state.company_name = company_name
                st.session_state.ticker = ticker
                st.session_state.conversation_history = []
                st.success("✓ 10-K filing loaded")

    st.divider()
    st.warning("""
    **Disclaimer**: This tool analyzes historical SEC 10-K filings for informational purposes only. Does NOT provide investment advice.
    """)

# Main content area with conditional tabs
if 'vectorstore' in st.session_state:
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Analytics Dashboard"])

    # ============ TAB 1: CHAT ============
    with tab1:
        st.header(f"{st.session_state.company_name}")

        with st.spinner("Fetching live market data..."):
            stock_info = get_stock_info(st.session_state.ticker)
            if stock_info:
                st.subheader("Live Market Data")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${stock_info['price']}")
                with col2:
                    st.metric("Day High", f"${stock_info['day_high']}")
                with col3:
                    market_cap = stock_info['market_cap']
                    if isinstance(market_cap, (int, float)):
                        st.metric("Market Cap", f"${market_cap:,}")
                    else:
                        st.metric("Market Cap", market_cap)
                with col4:
                    st.metric("P/E Ratio", stock_info['pe_ratio'])

        st.divider()
        st.subheader("Ask Questions About the 10-K Filing")

        def fetch_stock_info():
            if 'ticker' in st.session_state:
                return get_stock_info(st.session_state.ticker)
            return None

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Main Business Risks"):
                quick_question = "What are the main business risks?"
                st.session_state.conversation_history.append({"role": "user", "content": quick_question})
                with st.spinner("Analyzing..."):
                    answer = query_company(
                        st.session_state.vectorstore,
                        st.session_state.company_name,
                        quick_question,
                        stock_info=fetch_stock_info()
                    )
                    st.session_state.conversation_history.append({"role": "assistant", "content": answer})
                st.rerun()
        with col2:
            if st.button("Competitive Advantages"):
                quick_question = "What are the competitive advantages?"
                st.session_state.conversation_history.append({"role": "user", "content": quick_question})
                with st.spinner("Analyzing..."):
                    answer = query_company(
                        st.session_state.vectorstore,
                        st.session_state.company_name,
                        quick_question,
                        stock_info=fetch_stock_info()
                    )
                    st.session_state.conversation_history.append({"role": "assistant", "content": answer})
                st.rerun()
        with col3:
            if st.button("Revenue Sources"):
                quick_question = "How does the company generate revenue?"
                st.session_state.conversation_history.append({"role": "user", "content": quick_question})
                with st.spinner("Analyzing..."):
                    answer = query_company(
                        st.session_state.vectorstore,
                        st.session_state.company_name,
                        quick_question,
                        stock_info=fetch_stock_info()
                    )
                    st.session_state.conversation_history.append({"role": "assistant", "content": answer})
                st.rerun()

        st.divider()
        for msg in st.session_state.conversation_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input(f"Ask anything about {st.session_state.company_name}..."):
            st.session_state.conversation_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    answer = query_company(
                        st.session_state.vectorstore,
                        st.session_state.company_name,
                        prompt,
                        stock_info=fetch_stock_info()
                    )
                    st.write(answer)
            st.session_state.conversation_history.append({"role": "assistant", "content": answer})
            st.rerun()

        if st.session_state.conversation_history:
            st.divider()
            col1, col2 = st.columns([4, 1])
            with col2:
                conversation_json = json.dumps(st.session_state.conversation_history, indent=2)
                st.download_button(
                    label="📥 Export Chat",
                    data=conversation_json,
                    file_name=f"{st.session_state.company_name}_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    # ============ TAB 2: ANALYTICS ============
    with tab2:
        st.header("📊 Usage Analytics Dashboard")

        mlruns_path = Path('mlruns')

        if not mlruns_path.exists():
            st.warning("No analytics data available yet. Ask more questions to see metrics here.")
        else:
            runs_data = []

            with st.spinner("Loading analytics data..."):
                for experiment_dir in mlruns_path.iterdir():
                    if experiment_dir.is_dir() and experiment_dir.name not in ['0', '.trash']:
                        for run_dir in experiment_dir.glob('*/'):
                            if run_dir.is_dir():
                                metrics_dir = run_dir / 'metrics'
                                params_dir = run_dir / 'params'
                                if metrics_dir.exists() and params_dir.exists():
                                    run_info = {'run_id': run_dir.name}
                                    for param_file in params_dir.iterdir():
                                        if param_file.is_file():
                                            with open(param_file) as f:
                                                run_info[param_file.name] = f.read().strip()
                                    for metric_file in metrics_dir.iterdir():
                                        if metric_file.is_file():
                                            with open(metric_file) as f:
                                                content = f.read().strip().split()
                                                if len(content) >= 2:
                                                    try:
                                                        run_info[metric_file.name] = float(content[1])
                                                    except:
                                                        pass
                                    runs_data.append(run_info)

            if not runs_data:
                st.warning("No query data found yet. Ask questions to populate analytics.")
            else:
                df = pd.DataFrame(runs_data)

                # Summary Metrics
                st.subheader("📈 Summary Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Queries", len(df))
                with col2:
                    if 'estimated_cost_usd' in df.columns:
                        st.metric("Total API Cost", f"${df['estimated_cost_usd'].sum():.4f}")
                    else:
                        st.metric("Total API Cost", "N/A")
                with col3:
                    if 'latency_seconds' in df.columns:
                        st.metric("Avg Response Time", f"{df['latency_seconds'].mean():.2f}s")
                    else:
                        st.metric("Avg Response Time", "N/A")
                with col4:
                    if 'company' in df.columns:
                        st.metric("Companies Analyzed", df['company'].nunique())
                    else:
                        st.metric("Companies Analyzed", "N/A")
                st.divider()

                # ===================== COMPANY COMPARISON SECTION =====================
                companies = df['company'].unique().tolist() if 'company' in df.columns else []
                if len(companies) >= 2:
                    st.subheader("🏢 Company Comparison (Side-by-Side View)")
                    default_companies = companies[:2]
                    selected = st.multiselect(
                        "Select up to two companies for comparison:",
                        companies,
                        default=default_companies,
                        max_selections=2
                    )
                    if len(selected) == 2:
                        comp_df = df[df['company'].isin(selected)]
                        summary = comp_df.groupby('company').agg({
                            'estimated_cost_usd': 'sum',
                            'latency_seconds': 'mean',
                            'prompt_tokens': 'sum',
                            'completion_tokens': 'sum',
                            'response_length': 'mean'
                        }).reset_index()
                        summary_table = summary[['company', 'estimated_cost_usd', 'latency_seconds', 'prompt_tokens', 'completion_tokens', 'response_length']]
                        summary_table.columns = [
                            'Company',
                            'Total Cost (USD)',
                            'Avg Latency (s)',
                            'Input Tokens',
                            'Output Tokens',
                            'Avg Response Length'
                        ]
                        st.table(summary_table)
                    else:
                        st.info("Please select two companies for a side-by-side comparison.")
                else:
                    st.info("More than one company is needed for comparison to be available.")

                st.divider()

                # Cost Analysis
                if 'estimated_cost_usd' in df.columns:
                    st.subheader("💰 Cost Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Cumulative Cost Over Time**")
                        df['cumulative_cost'] = df['estimated_cost_usd'].cumsum()
                        st.line_chart(df[['cumulative_cost']])
                    with col2:
                        if 'company' in df.columns:
                            st.write("**Cost by Company**")
                            cost_by_company = df.groupby('company')['estimated_cost_usd'].sum().sort_values(ascending=False)
                            st.bar_chart(cost_by_company)

                st.divider()

                # Token Usage
                if 'prompt_tokens' in df.columns and 'completion_tokens' in df.columns:
                    st.subheader("🔢 Token Usage")
                    col1, col2 = st.columns(2)
                    with col1:
                        total_tokens = df['prompt_tokens'].sum() + df['completion_tokens'].sum()
                        st.metric("Total Tokens Used", f"{total_tokens:,}")
                        st.write(f"- Input tokens: {df['prompt_tokens'].sum():,}")
                        st.write(f"- Output tokens: {df['completion_tokens'].sum():,}")
                    with col2:
                        st.write("**Token Distribution**")
                        token_data = pd.DataFrame({
                            'Input': df['prompt_tokens'],
                            'Output': df['completion_tokens']
                        })
                        st.area_chart(token_data)

                st.divider()

                # Query Patterns
                if 'company' in df.columns:
                    st.subheader("📊 Query Patterns")
                    st.write("**Queries by Company**")
                    st.bar_chart(df['company'].value_counts())

                st.divider()

                # Performance Metrics
                if 'latency_seconds' in df.columns:
                    st.subheader("⚡ Performance Metrics")
                    st.write("**Response Time Over Time**")
                    st.line_chart(df['latency_seconds'])

                # Download Data
                st.divider()
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Analytics Data",
                    data=csv,
                    file_name=f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

elif Path('mlruns').exists():
    st.header("📊 Usage Analytics Dashboard")
    st.info("💡 Analyze a company from the sidebar to see chat interface, or view your usage analytics below.")
else:
    st.info("👈 Select a company from the sidebar to begin analysis")

    st.subheader("About This Tool")
    st.markdown("""
    This financial risk analyzer uses AI to help you understand SEC 10-K filings:
    - **Comprehensive Coverage**: Access 10,000+ publicly traded companies
    - **Intelligent Analysis**: Uses GPT-4o-mini to extract insights
    - **Real-time Market Data**: See current stock prices alongside risk analysis
    - **Usage Analytics**: Track your queries, costs, and performance metrics
    - **Conversation History**: Export your analysis sessions
    Built with LangChain, OpenAI, MLflow, and SEC EDGAR data.
    """)
