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

        # Generate raw response
        if self.last_query_type == "structured":
            raw_response = self._format_structured_response(retrieval_results, structured_files)
        else:
            raw_response = self._format_text_response(retrieval_results)

        # Apply the enhanced response formatting
        formatted_response = self.format_response(raw_response, query)
        
        # Add the response to conversation history
        self.conversation_history.append({"role": "assistant", "content": formatted_response})

        return formatted_response
    
    def format_response(self, raw_response, query):
        """Format the raw response to be more organized and readable."""
        # Identify query patterns for better formatting
        list_type_keywords = ['benefits', 'advantages', 'features', 'reasons', 'requirements']
        is_list_type = any(keyword in query.lower() for keyword in list_type_keywords)
        
        # Check if this is a definition query
        definition_query = query.lower().startswith(('what is', 'what are', 'who is', 'how does', 'define'))
        
        # Extract sources
        sources = []
        if 'Sources:' in raw_response:
            sources_section = raw_response.split('Sources:')[-1].strip()
            sources = [s.strip() for s in sources_section.split(',')]
        
        # Clean up the text by removing document artifacts
        cleaned_text = re.sub(r'From [^:]+\.pdf:', '', raw_response)
        cleaned_text = re.sub(r'\s*\.{3,}\s*', ' ', cleaned_text)  # Remove ellipses
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)     # Fix excessive newlines
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
        sentences = [s.strip() for s in sentences if 20 < len(s.strip()) < 300]  # Filter out overly short/long sentences
        
        # For definition queries: what is X?
        if definition_query:
            # First try finding explicit definitions with patterns like "X is..." or "X are..."
            query_terms = [term.lower() for term in query.split() if len(term) > 2 and term not in ['what', 'is', 'are', 'the', 'and', 'for', 'with']]
            
            # Try finding the most relevant definition
            definition_sentences = []
            for sentence in sentences:
                # Look for pattern: "X is/are Y" or similar
                if any(term in sentence.lower() for term in query_terms) and re.search(r'\b(is|are)\b', sentence.lower()):
                    # Give higher weight to sentences that explicitly answer the query
                    definition_sentences.append(sentence)
            
            if definition_sentences:
                # Use the first definition found
                response = definition_sentences[0]
                
                # Add one more relevant detail if available
                if len(definition_sentences) > 1:
                    response += f"\n\n{definition_sentences[1]}"
                
                # Add sources
                if sources:
                    response += f"\n\nSources: {', '.join(sources)}"
                
                return response
            
            # Fallback to most relevant sentences containing query terms
            relevant_sentences = []
            for sentence in sentences:
                if any(term in sentence.lower() for term in query_terms):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                response = relevant_sentences[0]
                if len(relevant_sentences) > 1:
                    response += f"\n\n{relevant_sentences[1]}"
                    
                # Add sources
                if sources:
                    response += f"\n\nSources: {', '.join(sources)}"
                    
                return response
        
        # For list-type queries: what are the benefits of X?
        if is_list_type or any(pattern in query.lower() for pattern in ['what are', 'tell me about', 'list', 'explain']):
            # Try to identify bullet points
            bullet_points = []
            
            # First look for explicit bullet patterns
            bullet_patterns = re.findall(r'(?:^|\n)(?:[-•*]\s*|\d+[.)\]]\s*)([^\n]+)', cleaned_text)
            if bullet_patterns:
                bullet_points = bullet_patterns
            else:
                # Extract sentences that could be list items from relevant parts
                query_terms = [term.lower() for term in query.split() if len(term) > 2 and term not in ['what', 'are', 'the', 'benefits', 'advantages', 'features', 'of']]
                
                for sentence in sentences:
                    # Only include relevant sentences that might be standalone points
                    if any(term in sentence.lower() for term in query_terms) and len(sentence) < 200:
                        # Avoid sentences that look like they're part of an explanation rather than a point
                        if not any(sentence.lower().startswith(word) for word in ['the', 'this', 'from', 'it', 'they']):
                            bullet_points.append(sentence)
            
            # Format and deduplicate the bullet points
            seen = set()
            unique_points = []
            
            for point in bullet_points:
                # Normalize for deduplication comparison
                point_key = re.sub(r'\W+', '', point.lower())[:30]
                
                if point_key not in seen and len(point) > 15:
                    seen.add(point_key)
                    
                    # Clean up the point
                    clean_point = re.sub(r'^[,;:\s]+', '', point)
                    
                    # Ensure proper capitalization and ending
                    if clean_point:
                        clean_point = clean_point[0].upper() + clean_point[1:]
                        if not clean_point.endswith(('.', '!', '?')):
                            clean_point += '.'
                        unique_points.append(clean_point)
            
            # Format as bullet list if we found valid points
            if unique_points:
                response = ""
                
                for point in unique_points:
                    response += f"• {point}\n"
                    
                # Add sources
                if sources:
                    response += f"\nSources: {', '.join(sources)}"
                    
                return response
        
        # For general queries - extract relevant paragraphs
        query_terms = [term.lower() for term in query.split() if len(term) > 2 and term not in ['what', 'is', 'are', 'the', 'and', 'for', 'with', 'about']]
        
        # Try to extract relevant paragraphs
        paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
        relevant_paragraphs = []
        
        for paragraph in paragraphs:
            if any(term in paragraph.lower() for term in query_terms):
                # Truncate very long paragraphs
                if len(paragraph) > 300:
                    # Find a good breaking point
                    for sentence_end in ['.', '!', '?']:
                        pos = paragraph.rfind(sentence_end, 0, 300)
                        if pos != -1:
                            paragraph = paragraph[:pos+1]
                            break
                
                # Don't add paragraphs that are just citations or headers
                if len(paragraph) > 30:
                    relevant_paragraphs.append(paragraph)
        
        # Format the response
        if relevant_paragraphs:
            # Use only the first 1-2 most relevant paragraphs
            response = relevant_paragraphs[0]
            
            # Add one more paragraph if available and not too long
            if len(relevant_paragraphs) > 1 and len(response) + len(relevant_paragraphs[1]) < 500:
                response += f"\n\n{relevant_paragraphs[1]}"
        else:
            # Fallback to the most relevant sentences if we couldn't find good paragraphs
            relevant_sentences = [s for s in sentences if any(term in s.lower() for term in query_terms)]
            
            if relevant_sentences:
                response = relevant_sentences[0]
                
                # Add a few more sentences if available
                additional = []
                for s in relevant_sentences[1:3]:
                    if s not in response:
                        additional.append(s)
                
                if additional:
                    response += "\n\n" + " ".join(additional)
            else:
                # Absolute fallback - just return a cleaned-up version of the first part of the text
                response = sentences[0] if sentences else "I couldn't find a clear answer to your question."
        
        # Add sources
        if sources:
            response += f"\n\nSources: {', '.join(sources)}"
        
        return response

    def _format_text_response(self, retrieval_results):
        """Format raw response for text-based queries."""
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
            # Join content from this source - preserve full content for formatting
            combined_text = "\n".join(texts)
            response += f"{combined_text}\n\n"

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
            
            # Display with proper line breaks
            print("\nRealEstateGPT:")
            for line in answer.split('\n'):
                print(f"  {line}")
            print()
            