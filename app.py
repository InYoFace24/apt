import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import os
from PIL import Image
from llm_integration import LLMApartmentAssistant

# Set up page configuration
st.set_page_config(
    page_title="Apt, Apt, huh, uh-huh uh-huh", 
    page_icon="🏡",
    layout="wide"
)

# Custom CSS - Fixed styling issues
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #2E86AB;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid;
}
.user-message {
    background-color: #E3F2FD;
    border-left-color: #1976D2;
    color: #000000 !important;  /* Force black text */
}
.assistant-message {
    background-color: #E8F5E9;
    border-left-color: #388E3C;
    color: #000000 !important;  /* Force black text */
}
.apartment-card {
    background: white;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #ddd;
    margin-bottom: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
}
.apartment-image {
    width: 100%;
    margin: 10px 0;
    border-radius: 5px;
}
/* Ensure all text in chat messages is black */
.chat-message * {
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# Display banner image
try:
    banner_image = Image.open("logo.jpg")
    st.image(banner_image, use_container_width=True, caption="Find Your Apt")
except FileNotFoundError:
    st.info("Banner image not found. Place an image named 'logo.jpg'.")

# Load data function
@st.cache_data
def load_apartment_data():
    """Load apartment data from the DaejeonGoKr.xlsx file."""
    try:
        # Load from the local Excel file
        df = pd.read_excel("DaejeonGoKr.xlsx")
        
        # Make sure the dataframe has all required columns
        required_columns = [
            'Property', 'Serial', 'Building', 'Room', 
            'Rental area', 'Net area', 'Common area',
            'Rental type', 'Deposit', 'Monthly rent'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Required column '{col}' missing from dataset")
                return pd.DataFrame()
                
        return df
        
    except FileNotFoundError:
        st.error("Data file 'DaejeonGoKr.xlsx' not found. Please make sure it's in the same directory as the app.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Natural language processing functions
def extract_price_range(query):
    """Extract price ranges from user queries."""
    query = query.lower()
    
    # Under pattern
    under_match = re.search(r'under\s*₩?(\d{1,3}(?:,\d{3})*)', query)
    if under_match:
        return (0, int(under_match.group(1).replace(',', '')))
    
    # Over pattern  
    over_match = re.search(r'over\s*₩?(\d{1,3}(?:,\d{3})*)', query)
    if over_match:
        return (int(over_match.group(1).replace(',', '')), float('inf'))
    
    # Between pattern
    between_match = re.search(r'between\s*₩?(\d{1,3}(?:,\d{3})*)\s*and\s*₩?(\d{1,3}(?:,\d{3})*)', query)
    if between_match:
        return (int(between_match.group(1).replace(',', '')), 
                int(between_match.group(2).replace(',', '')))
    
    return None

def extract_area_preference(query):
    """Extract area preferences from queries."""
    query = query.lower()
    
    # Area around X
    area_match = re.search(r'around\s*(\d+(?:\.\d+)?)\s*㎡?', query)
    if area_match:
        target_area = float(area_match.group(1))
        return (target_area - 3, target_area + 3)  # ±3㎡ range
    
    # Minimum area
    min_match = re.search(r'(?:at least|minimum|min)\s*(\d+(?:\.\d+)?)\s*㎡?', query)
    if min_match:
        return (float(min_match.group(1)), float('inf'))
    
    return None

def process_natural_query(query, df):
    """Process natural language queries and filter data."""
    query_lower = query.lower()
    filtered_df = df.copy()
    response_parts = []
    
    # Extract price range for monthly rent
    if 'rent' in query_lower or 'monthly' in query_lower:
        price_range = extract_price_range(query)
        if price_range:
            min_price, max_price = price_range
            if max_price != float('inf'):
                filtered_df = filtered_df[filtered_df['Monthly rent'] <= max_price]
                response_parts.append(f"Monthly rent under ₩{max_price:,}")
            else:
                filtered_df = filtered_df[filtered_df['Monthly rent'] >= min_price]
                response_parts.append(f"Monthly rent over ₩{min_price:,}")
    
    # Extract price range for deposit
    if 'deposit' in query_lower:
        price_range = extract_price_range(query)
        if price_range:
            min_price, max_price = price_range
            if max_price != float('inf'):
                filtered_df = filtered_df[filtered_df['Deposit'] <= max_price]
                response_parts.append(f"Deposit under ₩{max_price:,}")
            else:
                filtered_df = filtered_df[filtered_df['Deposit'] >= min_price]
                response_parts.append(f"Deposit over ₩{min_price:,}")
    
    # Extract area preferences
    area_range = extract_area_preference(query)
    if area_range:
        min_area, max_area = area_range
        if max_area != float('inf'):
            filtered_df = filtered_df[
                (filtered_df['Rental area'] >= min_area) & 
                (filtered_df['Rental area'] <= max_area)
            ]
            response_parts.append(f"Area between {min_area}㎡ and {max_area}㎡")
        else:
            filtered_df = filtered_df[filtered_df['Rental area'] >= min_area]
            response_parts.append(f"Area at least {min_area}㎡")
    
    # Filter by rental type keywords
    if 'youth' in query_lower or 'student' in query_lower:
        filtered_df = filtered_df[filtered_df['Rental type'].str.lower().str.contains('youth', na=False)]
        response_parts.append("Youth rentals")
    
    if '1st place' in query_lower or 'first place' in query_lower:
        filtered_df = filtered_df[filtered_df['Rental type'].str.lower().str.contains('1st place', na=False)]
        response_parts.append("1st place rentals")
    
    return filtered_df, response_parts

def format_apartment_response(df, query, response_parts):
    """Format the apartment search response with better handling of number of results."""
    if df.empty:
        return "I couldn't find any apartments matching your criteria. Try adjusting your search terms."
    
    query_lower = query.lower()
    response = ""
    
    # Extract number if user asks for specific count (e.g., "10 cheapest")
    count_match = re.search(r'(\d+)\s+(cheapest|largest|options)', query_lower)
    result_count = 5  # default number of results to show
    if count_match:
        result_count = min(int(count_match.group(1)), 20)  # cap at 20 for readability
    
    # Sort based on query intent
    if 'cheapest' in query_lower or 'lowest' in query_lower:
        df = df.sort_values('Monthly rent')
        if 'cheapest' in query_lower and not count_match:
            # If just asking for "cheapest" without number, show top 3 by default
            result_count = 3
        
        response += f"**Top {min(result_count, len(df))} cheapest options:**\n\n"
        for idx, row in df.head(result_count).iterrows():
            response += f"🏠 **{row['Property']}** - Room {row['Room']}\n"
            response += f"   💰 Monthly rent: ₩{row['Monthly rent']:,}\n"
            response += f"   💳 Deposit: ₩{row['Deposit']:,}\n"
            response += f"   📐 Area: {row['Rental area']}㎡ (Net: {row['Net area']}㎡)\n"
            response += f"   🏷️ Type: {row['Rental type']}\n\n"
        
    elif 'largest' in query_lower or 'biggest' in query_lower:
        df = df.sort_values('Rental area', ascending=False)
        if 'largest' in query_lower and not count_match:
            # If just asking for "largest" without number, show top 3 by default
            result_count = 3
        
        response += f"**Top {min(result_count, len(df))} largest options:**\n\n"
        for idx, row in df.head(result_count).iterrows():
            response += f"🏠 **{row['Property']}** - Room {row['Room']}\n"
            response += f"   📐 Area: {row['Rental area']}㎡ (Net: {row['Net area']}㎡)\n"
            response += f"   💰 Monthly rent: ₩{row['Monthly rent']:,}\n"
            response += f"   💳 Deposit: ₩{row['Deposit']:,}\n"
            response += f"   🏷️ Type: {row['Rental type']}\n\n"
            
    else:
        # General listing
        criteria = " | ".join(response_parts) if response_parts else "all apartments"
        response += f"**Found {len(df)} apartments** matching: {criteria}\n\n"
        
        # Show more results when filtered (up to 10)
        result_count = 10 if len(df) > 5 else len(df)
        response += f"**Showing top {result_count} results:**\n\n"
        
        for idx, row in df.head(result_count).iterrows():
            response += f"🏠 **{row['Property']}** - Room {row['Room']}\n"
            response += f"   📐 Area: {row['Rental area']}㎡ (Net: {row['Net area']}㎡)\n"
            response += f"   💰 Monthly: ₩{row['Monthly rent']:,} | Deposit: ₩{row['Deposit']:,}\n"
            response += f"   🏷️ Type: {row['Rental type']}\n\n"
        
        if len(df) > result_count:
            response += f"\n*There are {len(df) - result_count} more results. Use the filters to narrow down.*"
    
    return response

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'df' not in st.session_state:
    st.session_state.df = load_apartment_data()

if 'llm_assistant' not in st.session_state:
    st.session_state.llm_assistant = LLMApartmentAssistant()

def should_use_llm(query):
    """Determine if a query needs LLM processing"""
    simple_patterns = [
        r'\d+,\d+', r'\d+㎡', 'under', 'over', 
        'between', 'cheapest', 'largest'
    ]
    return not any(re.search(p, query.lower()) for p in simple_patterns)

# Header
st.markdown("<h1 class='main-header'>🏡 Find Your 'Apt' - Daejeon</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Your AI assistant for finding apartments in Daejeon, South Korea</p>", unsafe_allow_html=True)

# Sidebar for filters only (file upload removed)
with st.sidebar:
    st.header("🔍 Filters")
    
    df = st.session_state.df
    if not df.empty:
        # Price filters
        max_rent = st.slider(
            "Max Monthly Rent (₩)",
            min_value=int(df['Monthly rent'].min()),
            max_value=int(df['Monthly rent'].max()),
            value=int(df['Monthly rent'].max()),
            step=10000,
            format="₩%d"
        )
        
        max_deposit = st.slider(
            "Max Deposit (₩)",
            min_value=int(df['Deposit'].min()),
            max_value=int(df['Deposit'].max()),
            value=int(df['Deposit'].max()),
            step=100000,
            format="₩%d"
        )
        
        # Area filter
        area_range = st.slider(
            "Rental Area (㎡)",
            min_value=float(df['Rental area'].min()),
            max_value=float(df['Rental area'].max()),
            value=(float(df['Rental area'].min()), float(df['Rental area'].max())),
            step=0.1
        )
        
        # Rental type filter
        rental_types = df['Rental type'].unique()
        selected_types = st.multiselect(
            "Rental Types",
            options=rental_types,
            default=rental_types
        )
        
        # Apply filters
        filtered_df = df[
            (df['Monthly rent'] <= max_rent) &
            (df['Deposit'] <= max_deposit) &
            (df['Rental area'] >= area_range[0]) &
            (df['Rental area'] <= area_range[1]) &
            (df['Rental type'].isin(selected_types))
        ]
        
        st.info(f"Showing {len(filtered_df)} apartments")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📋 Browse Apartments", "📊 Market Stats"])

with tab1:
    st.header("Chat with Your Apartment Assistant")
    
    # Display chat history with images
    for chat in st.session_state.chat_history:
        # User message
        st.markdown("**You:**")
        st.info(chat['query'])
    
        # Assistant message  
        st.markdown("**Assistant:**")
        st.success(chat['response'])
    
        # Display images if available
        if 'images' in chat and chat['images']:
            st.markdown("**📸 Visual Examples:**")
        
            # Display images in columns
            if len(chat['images']) == 1:
                st.image(chat['images'][0], caption="Apartment Example", use_container_width=True)
            else:
                cols = st.columns(len(chat['images']))
                for i, img_path in enumerate(chat['images']):
                    with cols[i]:
                        try:
                            st.image(img_path, caption=f"Example {i+1}", use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not load image: {img_path}")
    
        st.markdown("---")  # Separator between conversations
    
    # Chat input
    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            user_query = st.text_input(
                "Ask about apartments:",
                placeholder="e.g., 'Show me youth rentals under ₩150,000' or 'Find apartments around 25㎡'",
                key="chat_input"
            )
        with col2:
            send_button = st.button("Send", type="primary", use_container_width=True)
    
    # Process query
    if send_button and user_query:
        with st.spinner("Searching..."):
            if should_use_llm(user_query):
                # LLM Path with images
                llm_response, docs, relevant_images = st.session_state.llm_assistant.rag_query(user_query)
        
                # Convert docs to DataFrame for compatibility with your UI
                result_df = pd.DataFrame([doc.metadata for doc in docs])
                response_parts = ["LLM-enhanced results"]
            else:
                # Original rule-based path (no images for now)
                result_df, response_parts = process_natural_query(user_query, st.session_state.df)
                llm_response = None
                relevant_images = []
    
            # Format using your existing UI system
            response = format_apartment_response(result_df, user_query, response_parts)
    
            # Combine responses if LLM was used
            if llm_response:
                response = f"{llm_response}\n\n---\n\n**Detailed Results:**\n\n{response}"
    
            st.session_state.chat_history.append({
                'query': user_query,
                'response': response,
                'images': relevant_images  # Store images with chat history
            })
            st.rerun()
    
    # Example queries
    if not st.session_state.chat_history:
        st.markdown("### 💡 Try asking:")
        example_queries = [
            "Show me youth rentals under ₩150,000",
            "Find apartments around 25㎡", 
            "What's the cheapest option available?",
            "Show me apartments with deposit under ₩1,500,000",
            "Find the largest apartments available"
        ]
        
        cols = st.columns(len(example_queries))
        for i, query in enumerate(example_queries):
            with cols[i]:
                if st.button(query, key=f"example_{query}", use_container_width=True):
                    # Set the query and trigger processing
                    result_df, criteria = process_natural_query(query, st.session_state.df)
                    response = format_apartment_response(result_df, query, criteria)
                    
                    st.session_state.chat_history.append({
                        'query': query,
                        'response': response
                    })
                    
                    st.rerun()

with tab2:
    st.header("Browse All Apartments")
    
    if not df.empty:
        # Display filtered results
        st.dataframe(
            filtered_df.style.format({
                'Monthly rent': '₩{:,}',
                'Deposit': '₩{:,}',
                'Rental area': '{:.1f}㎡',
                'Net area': '{:.1f}㎡',
                'Common area': '{:.1f}㎡'
            }),
            use_container_width=True,
            height=400
        )
        
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name=f"daejeon_apartments_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with tab3:
    st.header("Market Statistics")
    
    if not df.empty:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Apartments</h3>
                <h2>{}</h2>
            </div>
            """.format(len(filtered_df)), unsafe_allow_html=True)
        
        with col2:
            avg_rent = filtered_df['Monthly rent'].mean()
            st.markdown("""
            <div class="metric-card">
                <h3>Avg Monthly Rent</h3>
                <h2>₩{:,}</h2>
            </div>
            """.format(int(avg_rent)), unsafe_allow_html=True)
        
        with col3:
            avg_deposit = filtered_df['Deposit'].mean()
            st.markdown("""
            <div class="metric-card">
                <h3>Avg Deposit</h3>
                <h2>₩{:,}</h2>
            </div>
            """.format(int(avg_deposit)), unsafe_allow_html=True)
        
        with col4:
            avg_area = filtered_df['Rental area'].mean()
            st.markdown("""
            <div class="metric-card">
                <h3>Avg Area</h3>
                <h2>{:.1f}㎡</h2>
            </div>
            """.format(avg_area), unsafe_allow_html=True)
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution
            fig_price = px.histogram(
                filtered_df, 
                x='Monthly rent',
                nbins=20,
                title='Monthly Rent Distribution',
                color_discrete_sequence=['#1f77b4']
            )
            fig_price.update_layout(
                xaxis_title="Monthly Rent (₩)",
                yaxis_title="Number of Apartments"
            )
            st.plotly_chart(fig_price, use_container_width=True)
        
        with col2:
            # Area vs Price scatter
            fig_scatter = px.scatter(
                filtered_df,
                x='Rental area',
                y='Monthly rent',
                color='Rental type',
                title='Area vs Monthly Rent',
                hover_data=['Deposit', 'Room']
            )
            fig_scatter.update_layout(
                xaxis_title="Rental Area (㎡)",
                yaxis_title="Monthly Rent (₩)"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Rental type breakdown
        rental_type_counts = filtered_df['Rental type'].value_counts()
        fig_pie = px.pie(
            values=rental_type_counts.values,
            names=rental_type_counts.index,
            title='Rental Type Distribution'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**💡 Tips:**")
st.markdown("""
- Use natural language: "Show me apartments under ₩150,000"
- Filter by area: "Find apartments around 25㎡" 
- Compare options: "What's the cheapest/largest apartment?"
""")

#streamlit run app.py