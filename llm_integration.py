from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import pandas as pd
import os
import glob
from typing import List, Dict, Tuple
import random

class LLMApartmentAssistant:
    def __init__(self, data_path="DaejeonGoKr.xlsx", images_folder="apartment_images"):
        self.df = pd.read_excel(data_path)
        self.images_folder = images_folder
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = self._create_vectorstore()
        self.llm = Ollama(model="llama3", temperature=0.3)
        self.image_mappings = self._create_image_mappings()
        
    def _create_vectorstore(self):
        # Convert DataFrame to LangChain Documents
        docs = []
        for _, row in self.df.iterrows():
            content = f"""
            Property: {row['Property']}
            Serial: {row['Serial']}
            Building: {row['Building']}
            Room: {row['Room']}
            Rent: ₩{row['Monthly rent']:,}
            Deposit: ₩{row['Deposit']:,}
            Area: {row['Rental area']}㎡
            Type: {row['Rental type']}
            """
            docs.append(Document(page_content=content, metadata=row.to_dict()))
        
        return Chroma.from_documents(docs, self.embeddings)
    
    def _create_image_mappings(self):
        """Create mappings between apartment properties and available images"""
        if not os.path.exists(self.images_folder):
            return {}
        
        image_mappings = {}
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.images_folder, ext)))
        
        # Create mappings based on filename patterns
        for image_path in image_files:
            filename = os.path.basename(image_path).lower()
            
            # Map by rental type
            if 'youth' in filename or 'student' in filename:
                if 'youth' not in image_mappings:
                    image_mappings['youth'] = []
                image_mappings['youth'].append(image_path)
            
            if 'apartment' in filename or 'flat' in filename:
                if 'apartment' not in image_mappings:
                    image_mappings['apartment'] = []
                image_mappings['apartment'].append(image_path)
            
            if 'modern' in filename or 'luxury' in filename:
                if 'modern' not in image_mappings:
                    image_mappings['modern'] = []
                image_mappings['modern'].append(image_path)
            
            # Map by property name (extract property names from your data)
            property_names = self.df['Property'].unique()
            for prop_name in property_names:
                if prop_name.lower().replace(' ', '').replace('-', '') in filename.replace(' ', '').replace('-', ''):
                    if prop_name not in image_mappings:
                        image_mappings[prop_name] = []
                    image_mappings[prop_name].append(image_path)
            
            # Generic apartment images
            if 'generic' in filename or 'sample' in filename:
                if 'generic' not in image_mappings:
                    image_mappings['generic'] = []
                image_mappings['generic'].append(image_path)
        
        return image_mappings
    
    def _find_relevant_images(self, relevant_docs: List[Document], query: str) -> List[str]:
        """Find images that match the apartment results"""
        relevant_images = []
        
        # Check if we have apartment-specific images
        for doc in relevant_docs:
            property_name = doc.metadata.get('Property', '')
            rental_type = doc.metadata.get('Rental type', '').lower()
            
            # Try to find images by property name
            if property_name in self.image_mappings:
                relevant_images.extend(self.image_mappings[property_name])
            
            # Try to find images by rental type
            for mapping_key in self.image_mappings:
                if mapping_key.lower() in rental_type:
                    relevant_images.extend(self.image_mappings[mapping_key])
        
        # If no specific images found, check query keywords
        if not relevant_images:
            query_lower = query.lower()
            for mapping_key in self.image_mappings:
                if mapping_key.lower() in query_lower:
                    relevant_images.extend(self.image_mappings[mapping_key])
        
        # If still no images, use generic ones
        if not relevant_images and 'generic' in self.image_mappings:
            relevant_images.extend(self.image_mappings['generic'])
        
        # Remove duplicates and limit to 3 images max
        relevant_images = list(set(relevant_images))[:3]
        
        return relevant_images
    
    def rag_query(self, query: str) -> Tuple[str, List[Document], List[str]]:
        """Enhanced RAG query that returns both text response and relevant images"""
        # Enhanced natural language processing
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a friendly and helpful assistant for international students looking for apartments in Daejeon, South Korea. 

Your goal is to help students find suitable housing based on their needs and preferences.

When responding to questions:
1. Be warm and conversational, like a helpful friend who knows the area well
2. If asked about multiple apartments, provide a list of options with key details
3. Try to offer additional context about neighborhoods when relevant
4. Make recommendations based on the student's apparent needs
5. When pricing is mentioned, always specify if it's monthly or deposit
6. If images are available, mention that you're including visual examples

Use the following apartment information to answer the user's questions:
{context}

Question: {question}

If you don't know the answer, say so politely and offer to help with other aspects of apartment hunting.
"""
        )
        
        # Retrieve relevant apartments
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        relevant_docs = retriever.get_relevant_documents(query)
        
        # Find relevant images
        relevant_images = self._find_relevant_images(relevant_docs, query)
        
        # Generate LLM response
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Modify the prompt to mention images if available
        if relevant_images:
            modified_context = context + f"\n\nNote: {len(relevant_images)} relevant image(s) are available to show the user."
            response = self.llm(prompt.format(context=modified_context, question=query))
        else:
            response = self.llm(prompt.format(context=context, question=query))
        
        return response, relevant_docs, relevant_images