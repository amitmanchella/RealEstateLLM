import pandas as pd
import numpy as np
import os
import re
import pickle
from enhanced_rag import EnhancedRealEstateRAG


class RealEstateChatbot:
    def __init__(self, rag_system=None):
        """
        Initialize the chatbot with a RAG system.

        Args:
            rag_system: An initialized EnhancedRealEstateRAG object
        """
        # Initialize RAG if not provided
        if rag_system is None:
            print("Initializing new RAG system...")
            self.rag = EnhancedRealEstateRAG()

            # Load embeddings and build index
            if os.path.exists("text_embeddings.pkl"):
                self.rag.load_embeddings()
                self.rag.build_indices()
            else:
                print("No embeddings found. Please create embeddings first.")
                return
        else:
            self.rag = rag_system

        self.conversation_history = []
        self.last_query_type = None
        self.last_structured_data_file = None

    def answer(self, query, k=5, show_context=False, query_type=None):
        """
        Answer a user question by retrieving relevant context.

        Args:
            query: The user's question
            k: Number of contexts to retrieve
            show_context: Whether to display the retrieved context
            query_type: Force a specific query type ("text", "structured", or "both")

        Returns:
            A formatted response
        """
        # Save the user's query to conversation history
        self.conversation_history.append({"role": "user", "content": query})

        # Check for special commands
        if query.lower().startswith("analyze "):
            return self._handle_analysis_command(query[8:])

        # Retrieve relevant information
        retrieval_results = self.rag.answer_question(query, k=k)
        context = retrieval_results['context']
        self.last_query_type = retrieval_results['query_type']

        # Show the retrieved context if requested
        if show_context:
            print("RETRIEVED CONTEXT:")
            print("-" * 80)
            print(context)
            print("-" * 80)
            print()

        # Get sources for citation
        sources = []
        structured_files = []
        for item in retrieval_results['retrieved_content']:
            source = item['source_file']
            if source not in sources:
                sources.append(source)

            # Track structured data files for follow-up
            if item.get('type') in ['structured', 'structured_column']:
                if source not in structured_files:
                    structured_files.append(source)
                    self.last_structured_data_file = source

        # Format response based on query type
        if self.last_query_type == "structured":
            response = self._format_structured_response(retrieval_results, structured_files)
        else:
            response = self._format_text_response(retrieval_results)

        # Add the response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def _format_text_response(self, retrieval_results):
        """Format response for text-based queries."""
        response = "Based on the real estate documents I've analyzed:\n\n"

        # Group by source file for better organization
        by_source = {}
        for item in retrieval_results['retrieved_content']:
            source = item['source_file']
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(item['text'])

        # Build response from each source
        for source, texts in by_source.items():
            response += f"From {source}:\n"

            # Join and truncate content from this source
            combined_text = "\n".join(texts)
            truncated = combined_text[:500] + "..." if len(combined_text) > 500 else combined_text
            response += f"{truncated}\n\n"

        # Add sources
        sources = list(by_source.keys())
        response += f"Sources: {', '.join(sources)}"

        return response

    def _format_structured_response(self, retrieval_results, structured_files):
        """Format response for structured data queries."""
        response = "Based on the real estate data analysis:\n\n"

        for item in retrieval_results['retrieved_content']:
            if item.get('type') in ['structured', 'structured_column']:
                response += f"From {item['source_file']}:\n"
                response += f"{item['text']}\n\n"

        # Add follow-up hint
        if structured_files:
            response += f"You can ask me to analyze specific aspects of the {structured_files[0]} dataset. "
            response += f"For example, try 'analyze {structured_files[0]} by price range'.\n\n"

        return response

    def _handle_analysis_command(self, command):
        """Handle special analysis commands for structured data."""
        # Parse command to extract file and analysis type
        parts = command.strip().split()
        if len(parts) < 2:
            return "Please specify what to analyze. For example: 'analyze dataset.csv by price'"

        filename = parts[0]
        analysis_type = " ".join(parts[1:])

        # Find the structured dataset
        if not self.rag.structured_data:
            return "No structured data is available for analysis."

        dataset = None
        for item in self.rag.structured_data:
            if item['filename'] == filename:
                dataset = item['data']
                break

        if dataset is None:
            return f"Dataset '{filename}' not found. Available datasets: " + \
                ", ".join([item['filename'] for item in self.rag.structured_data])

        # Perform different analyses based on the command
        try:
            if "summary" in analysis_type.lower():
                return self._generate_summary_stats(dataset, filename)
            elif "price range" in analysis_type.lower() or "price distribution" in analysis_type.lower():
                return self._analyze_price_distribution(dataset, filename)
            elif "location" in analysis_type.lower() or "area" in analysis_type.lower():
                return self._analyze_by_location(dataset, filename)
            elif "trend" in analysis_type.lower() or "time" in analysis_type.lower():
                return self._analyze_time_trends(dataset, filename)
            else:
                return f"I'm not sure how to analyze {filename} by {analysis_type}. " + \
                    "Try asking for a summary, price range, location analysis, or time trends."
        except Exception as e:
            return f"Error analyzing {filename}: {str(e)}"

    def _generate_summary_stats(self, df, filename):
        """Generate summary statistics for a dataset."""
        response = f"## Summary Statistics for {filename}\n\n"

        # Basic dataset info
        response += f"This dataset contains {len(df)} properties with {len(df.columns)} attributes.\n\n"

        # Try to identify numeric columns for statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            response += "### Key Metrics\n\n"

            # Look for price/value columns
            price_cols = [col for col in numeric_cols
                          if any(term in col.lower() for term in ['price', 'value', 'cost', 'amount'])]

            if price_cols:
                for col in price_cols[:2]:  # Limit to first 2 price columns
                    response += f"**{col}**:\n"
                    response += f"- Average: ${df[col].mean():.2f}\n"
                    response += f"- Median: ${df[col].median():.2f}\n"
                    response += f"- Range: ${df[col].min():.2f} to ${df[col].max():.2f}\n\n"

            # Look for size/area columns
            size_cols = [col for col in numeric_cols
                         if any(term in col.lower() for term in ['size', 'area', 'sqft', 'feet', 'acre'])]

            if size_cols:
                for col in size_cols[:2]:  # Limit to first 2 size columns
                    response += f"**{col}**:\n"
                    response += f"- Average: {df[col].mean():.2f}\n"
                    response += f"- Median: {df[col].median():.2f}\n"
                    response += f"- Range: {df[col].min():.2f} to {df[col].max():.2f}\n\n"

        # Try to identify categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            # Look for location columns
            location_cols = [col for col in categorical_cols
                             if any(term in col.lower() for term in ['city', 'state', 'zip', 'location', 'address'])]

            if location_cols:
                for col in location_cols[:1]:  # Limit to first location column
                    top_locations = df[col].value_counts().nlargest(5)
                    response += f"**Top {col}**:\n"
                    for loc, count in top_locations.items():
                        response += f"- {loc}: {count} properties ({count / len(df) * 100:.1f}%)\n"
                    response += "\n"

            # Look for property type columns
            type_cols = [col for col in categorical_cols
                         if any(term in col.lower() for term in ['type', 'style', 'property', 'category'])]

            if type_cols:
                for col in type_cols[:1]:  # Limit to first type column
                    top_types = df[col].value_counts().nlargest(5)
                    response += f"**Top {col}**:\n"
                    for typ, count in top_types.items():
                        response += f"- {typ}: {count} properties ({count / len(df) * 100:.1f}%)\n"
                    response += "\n"

        response += "You can ask for more specific analyses like 'analyze " + \
                    f"{filename} by price range' or 'analyze {filename} by location'"

        return response

    def _analyze_price_distribution(self, df, filename):
        """Analyze price distribution in the dataset."""
        # Try to find price column
        price_cols = [col for col in df.columns
                      if any(term in col.lower() for term in ['price', 'value', 'cost', 'amount'])]

        if not price_cols:
            return f"Could not identify a price column in {filename}."

        price_col = price_cols[0]  # Use the first price column found

        # Ensure the column is numeric
        if not pd.api.types.is_numeric_dtype(df[price_col]):
            return f"The column {price_col} is not numeric, cannot analyze price distribution."

        response = f"## Price Distribution Analysis for {filename}\n\n"

        # Calculate price ranges
        min_price = df[price_col].min()
        max_price = df[price_col].max()

        response += f"Price range from ${min_price:.2f} to ${max_price:.2f}\n\n"

        # Create price brackets
        brackets = 5
        bracket_size = (max_price - min_price) / brackets

        response += "### Price Brackets\n\n"
        for i in range(brackets):
            lower = min_price + i * bracket_size
            upper = lower + bracket_size
            count = df[(df[price_col] >= lower) & (df[price_col] < upper)].shape[0]
            percentage = count / len(df) * 100

            response += f"- ${lower:.2f} to ${upper:.2f}: {count} properties ({percentage:.1f}%)\n"

        # Properties above the highest bracket
        count = df[df[price_col] >= max_price].shape[0]
        percentage = count / len(df) * 100
        response += f"- ${max_price:.2f} and above: {count} properties ({percentage:.1f}%)\n\n"

        return response

    def _analyze_by_location(self, df, filename):
        """Analyze properties by location."""
        # Try to find location columns
        location_cols = [col for col in df.columns
                         if any(term in col.lower() for term in ['city', 'state', 'zip', 'location', 'address'])]

        if not location_cols:
            return f"Could not identify a location column in {filename}."

        location_col = location_cols[0]  # Use the first location column found

        response = f"## Location Analysis for {filename}\n\n"

        # Count properties by location
        location_counts = df[location_col].value_counts().nlargest(10)

        response += "### Top 10 Locations\n\n"
        for location, count in location_counts.items():
            percentage = count / len(df) * 100
            response += f"- {location}: {count} properties ({percentage:.1f}%)\n"

        # Try to find price column to analyze price by location
        price_cols = [col for col in df.columns
                      if any(term in col.lower() for term in ['price', 'value', 'cost', 'amount'])]

        if price_cols and pd.api.types.is_numeric_dtype(df[price_cols[0]]):
            price_col = price_cols[0]
            response += f"\n### Average Prices by Top 5 Locations\n\n"

            for location in location_counts.index[:5]:
                avg_price = df[df[location_col] == location][price_col].mean()
                response += f"- {location}: ${avg_price:.2f}\n"

        return response

    def _analyze_time_trends(self, df, filename):
        """Analyze trends over time."""
        # Try to find date columns
        date_cols = [col for col in df.columns
                     if any(term in col.lower() for term in ['date', 'year', 'month', 'time'])]

        if not date_cols:
            return f"Could not identify a date/time column in {filename}."

        date_col = date_cols[0]  # Use the first date column found

        # Try to convert to datetime if not already
        try:
            if not pd.api.types.is_datetime64_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        except:
            return f"Could not convert {date_col} to a valid date format."

        # Drop rows with invalid dates
        df = df.dropna(subset=[date_col])

        response = f"## Time Trend Analysis for {filename}\n\n"

        # Extract year and month
        df['year'] = df[date_col].dt.year

        # Count by year
        year_counts = df['year'].value_counts().sort_index()

        response += "### Properties by Year\n\n"
        for year, count in year_counts.items():
            percentage = count / len(df) * 100
            response += f"- {year}: {count} properties ({percentage:.1f}%)\n"

        # Try to find price column to analyze price trends
        price_cols = [col for col in df.columns
                      if any(term in col.lower() for term in ['price', 'value', 'cost', 'amount'])]

        if price_cols and pd.api.types.is_numeric_dtype(df[price_cols[0]]):
            price_col = price_cols[0]
            response += f"\n### Average Prices by Year\n\n"

            price_by_year = df.groupby('year')[price_col].mean()

            for year, avg_price in price_by_year.items():
                response += f"- {year}: ${avg_price:.2f}\n"

            # Calculate price change between first and last year
            if len(price_by_year) > 1:
                first_year = price_by_year.index[0]
                last_year = price_by_year.index[-1]
                price_change = ((price_by_year[last_year] / price_by_year[first_year]) - 1) * 100

                response += f"\nPrice change from {first_year} to {last_year}: {price_change:.1f}%\n"

        return response

    def chat(self):
        """Start an interactive chat session."""
        print("Welcome to RealEstateGPT! Ask me anything about real estate.")
        print("Type 'exit' to end the conversation.\n")

        while True:
            query = input("You: ")
            if query.lower() in ['exit', 'quit', 'bye']:
                print("RealEstateGPT: Goodbye! Hope I was helpful.")
                break

            answer = self.answer(query, show_context=False)
            print(f"\nRealEstateGPT: {answer}\n")