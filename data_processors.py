import pandas as pd
import docx
import PyPDF2
import os
from tqdm import tqdm
import pickle


class DataProcessor:
    """Base class for processing different types of real estate data"""

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def save_processed_data(self, data, output_path):
        """Save processed data to a pickle file"""
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved processed data to {output_path}")


class StructuredDataProcessor(DataProcessor):
    """Process structured data (Excel/CSV files)"""

    def process_excel(self, file_path):
        """Process Excel file and return structured data"""
        try:
            df = pd.read_excel(file_path)
            return {
                'filename': os.path.basename(file_path),
                'data': df,
                'metadata': {
                    'columns': df.columns.tolist(),
                    'rows': len(df),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
                }
            }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def process_csv(self, file_path):
        """Process CSV file and return structured data"""
        try:
            df = pd.read_csv(file_path)
            return {
                'filename': os.path.basename(file_path),
                'data': df,
                'metadata': {
                    'columns': df.columns.tolist(),
                    'rows': len(df),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
                }
            }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def process_all(self):
        """Process all structured data files in the data directory"""
        results = []

        # Find all Excel and CSV files
        excel_files = [f for f in os.listdir(self.data_dir)
                       if f.lower().endswith(('.xlsx', '.xls'))]
        csv_files = [f for f in os.listdir(self.data_dir)
                     if f.lower().endswith('.csv')]

        # Process Excel files
        for file in tqdm(excel_files, desc="Processing Excel files"):
            file_path = os.path.join(self.data_dir, file)
            result = self.process_excel(file_path)
            if result:
                results.append(result)

        # Process CSV files
        for file in tqdm(csv_files, desc="Processing CSV files"):
            file_path = os.path.join(self.data_dir, file)
            result = self.process_csv(file_path)
            if result:
                results.append(result)

        return results


class DocProcessor(DataProcessor):
    """Process Word documents"""

    def extract_text_from_docx(self, file_path):
        """Extract text from a Word document"""
        try:
            doc = docx.Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return '\n'.join(full_text)
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return ""

    def process_all(self):
        """Process all Word documents in the data directory"""
        results = []

        # Find all Word documents
        docx_files = [f for f in os.listdir(self.data_dir)
                      if f.lower().endswith('.docx')]

        # Process Word documents
        for file in tqdm(docx_files, desc="Processing Word documents"):
            file_path = os.path.join(self.data_dir, file)
            try:
                text = self.extract_text_from_docx(file_path)
                results.append({
                    'filename': file,
                    'text': text,
                    'size': os.path.getsize(file_path)
                })
                print(f"Successfully processed {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

        return results