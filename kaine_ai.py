#!/usr/bin/env python3
"""
Kaine AI - Telegram Posts Q&A Tool using OpenAI API
This tool includes:
- Post chunking for granular embeddings
- Hybrid search (semantic + keyword via TF-IDF)
- Optional context compression (disabled by default for cost efficiency)
- Configurable compression threshold via environment variables
- Cost tracking and estimation
- Diversity reranking for better post selection
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
from openai import OpenAI
import tiktoken
import pickle
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
from math import log

# Load environment variables from .env file
load_dotenv()

class TelegramQA:
    def __init__(self, json_file: str, api_key: str = None):
        """
        Initialize the Telegram Q&A tool
        
        Args:
            json_file: Path to the JSON file containing Telegram posts
            api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable or .env file")
        
        self.client = OpenAI(api_key=self.api_key)
        self.json_file = json_file
        self.posts = []  # Original posts
        self.chunks = []  # List of chunks with metadata
        self.embeddings = []

        # Ensure cache directory exists
        cache_dir = Path('cache')
        cache_dir.mkdir(exist_ok=True)

        # Cache files stored in cache/ directory
        self.embeddings_file = cache_dir / (Path(json_file).stem + '_embeddings.pkl')
        self.tfidf_file = cache_dir / (Path(json_file).stem + '_tfidf.pkl')
        self.chunks_file = cache_dir / (Path(json_file).stem + '_chunks.pkl')

        # Cost tracking
        self.total_cost = 0.0
        self.enable_cost_tracking = os.getenv('ENABLE_COST_TRACKING', 'false').lower() == 'true'
        
        # Load model configurations from environment
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')  # Upgraded to large for better quality
        self.chat_model = os.getenv('CHAT_MODEL', 'gpt-4o')  # Updated to gpt-4o for efficiency
        self.default_context_posts = int(os.getenv('DEFAULT_CONTEXT_POSTS', '15'))  # Increased default for more context
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '512'))  # Tokens per chunk
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '50'))

        # Compression settings (disabled by default for cost efficiency)
        self.enable_compression = os.getenv('ENABLE_COMPRESSION', 'false').lower() == 'true'
        self.compression_threshold = int(os.getenv('COMPRESSION_THRESHOLD', '100000'))  # Very high default

        # Diversity reranking
        self.enable_diversity = os.getenv('ENABLE_DIVERSITY', 'true').lower() == 'true'
        
        # Tokenizer for chunking
        self.tokenizer = tiktoken.encoding_for_model(self.embedding_model)
        
        # Load the posts
        self.load_posts()
        
        # Load or create chunks, embeddings, and TF-IDF
        self.load_or_create_chunks()
        self.load_or_create_embeddings()
        self.load_or_create_tfidf()
    
    def load_posts(self):
        """Load posts from JSON file"""
        print(f"Loading posts from {self.json_file}...")
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.posts = data['posts']
        print(f"Loaded {len(self.posts)} posts")
    
    def _create_searchable_text(self, post: Dict) -> str:
        """Create a searchable text representation of a post (normalized)"""
        parts = []
        
        # Add date
        if post.get('date'):
            parts.append(f"Date: {post['date']}")
        
        # Add user info
        if post.get('user'):
            user = post['user']
            username = user.get('username', 'Unknown').lower()  # Normalize
            first_name = user.get('first_name', '').lower()
            parts.append(f"User: {first_name} (@{username})")
        
        # Add content
        if post.get('content'):
            parts.append(f"Content: {post['content'].lower()}")  # Normalize for search
        
        # Add media info if present
        if post.get('media'):
            media = post['media']
            if media.get('type'):
                parts.append(f"Media: {media['type'].lower()}")
            if media.get('description'):
                parts.append(f"Media Description: {media['description'].lower()}")
        
        return "\n".join(parts).strip()  # Strip excess whitespace
    
    def load_or_create_chunks(self):
        """Create or load chunks from posts for granular embeddings"""
        if os.path.exists(self.chunks_file):
            print(f"Loading existing chunks from {self.chunks_file}...")
            with open(self.chunks_file, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"Loaded {len(self.chunks)} chunks")
        else:
            print("Creating chunks from posts...")
            self.create_chunks()
            with open(self.chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            print("Chunks saved")
    
    def create_chunks(self):
        """Split posts into chunks for better granularity"""
        self.chunks = []
        for post_idx, post in enumerate(self.posts):
            text = self._create_searchable_text(post)
            tokens = self.tokenizer.encode(text)
            
            if len(tokens) <= self.chunk_size:
                self.chunks.append({
                    'post_idx': post_idx,
                    'chunk_idx': 0,
                    'text': text,
                    'metadata': {
                        'date': post.get('date'),
                        'user': post.get('user', {}),
                        'telegram_url': post.get('telegram_url')
                    }
                })
            else:
                # Split with overlap
                for start in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
                    end = start + self.chunk_size
                    chunk_tokens = tokens[start:end]
                    chunk_text = self.tokenizer.decode(chunk_tokens)
                    self.chunks.append({
                        'post_idx': post_idx,
                        'chunk_idx': len(self.chunks),
                        'text': chunk_text,
                        'metadata': {
                            'date': post.get('date'),
                            'user': post.get('user', {}),
                            'telegram_url': post.get('telegram_url')
                        }
                    })
                    if end >= len(tokens):
                        break
        print(f"Created {len(self.chunks)} chunks")
    
    def load_or_create_embeddings(self):
        """Load existing embeddings or create new ones for chunks"""
        if os.path.exists(self.embeddings_file):
            print(f"Loading existing embeddings from {self.embeddings_file}...")
            with open(self.embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            print(f"Loaded {len(self.embeddings)} embeddings")
        else:
            print("Creating embeddings for chunks (this may take a while)...")
            self.create_embeddings()
            self.save_embeddings()
    
    def create_embeddings(self):
        """Create embeddings for all chunks using OpenAI API"""
        self.embeddings = []
        batch_size = 100
        
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i+batch_size]
            texts = [chunk['text'] for chunk in batch]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(self.chunks)-1)//batch_size + 1}...")
            
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )
                for embedding in response.data:
                    vec = np.array(embedding.embedding)
                    vec /= np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else 1  # Normalize
                    self.embeddings.append(vec)
            except Exception as e:
                print(f"Error creating embeddings: {e}")
                for _ in batch:
                    self.embeddings.append(np.zeros(3072))  # Size for text-embedding-3-large
        
        print(f"Created {len(self.embeddings)} embeddings")
    
    def save_embeddings(self):
        """Save embeddings to file"""
        print(f"Saving embeddings to {self.embeddings_file}...")
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        print("Embeddings saved")

    def track_cost(self, input_tokens: int, output_tokens: int, model: str):
        """Track API costs"""
        if not self.enable_cost_tracking:
            return

        # Pricing as of Jan 2025 (per 1M tokens)
        pricing = {
            'gpt-4o': {'input': 2.50, 'output': 10.00},
            'gpt-4-turbo-preview': {'input': 10.00, 'output': 30.00},
            'text-embedding-3-large': {'input': 0.13, 'output': 0.0},
            'text-embedding-3-small': {'input': 0.02, 'output': 0.0},
        }

        if model in pricing:
            cost = (input_tokens / 1_000_000 * pricing[model]['input'] +
                   output_tokens / 1_000_000 * pricing[model]['output'])
            self.total_cost += cost
            if self.enable_cost_tracking:
                print(f"[Cost] {model}: ${cost:.4f} (Total: ${self.total_cost:.4f})")
    
    def load_or_create_tfidf(self):
        """Load or create TF-IDF vectors for keyword search"""
        if os.path.exists(self.tfidf_file):
            print(f"Loading TF-IDF from {self.tfidf_file}...")
            with open(self.tfidf_file, 'rb') as f:
                self.term_to_id, self.idf, self.chunk_tfs = pickle.load(f)
            print("Loaded TF-IDF")
        else:
            print("Creating TF-IDF for hybrid search...")
            self.create_tfidf()
            with open(self.tfidf_file, 'wb') as f:
                pickle.dump((self.term_to_id, self.idf, self.chunk_tfs), f)
            print("TF-IDF saved")
    
    def create_tfidf(self):
        """Simple TF-IDF implementation using numpy"""
        # Build vocabulary
        all_terms = set()
        for chunk in self.chunks:
            terms = chunk['text'].lower().split()  # Simple tokenization
            all_terms.update(terms)
        self.term_to_id = {term: idx for idx, term in enumerate(sorted(all_terms))}
        vocab_size = len(self.term_to_id)
        
        # Document frequency
        df = np.zeros(vocab_size)
        for chunk in self.chunks:
            terms = set(chunk['text'].lower().split())
            for term in terms:
                df[self.term_to_id[term]] += 1
        
        # IDF
        self.idf = np.log(len(self.chunks) / (df + 1))  # Smoothing
        
        # TF per chunk
        self.chunk_tfs = []
        for chunk in self.chunks:
            tf = np.zeros(vocab_size)
            terms = chunk['text'].lower().split()
            for term in terms:
                tid = self.term_to_id.get(term)
                if tid is not None:
                    tf[tid] += 1
            tf = tf / (len(terms) + 1)  # Normalized TF
            self.chunk_tfs.append(tf)
    
    def compute_tfidf_scores(self, query_terms: List[str]) -> List[float]:
        """Compute TF-IDF scores for a query"""
        query_tf = np.zeros(len(self.term_to_id))
        for term in query_terms:
            tid = self.term_to_id.get(term)
            if tid is not None:
                query_tf[tid] += 1
        query_tf /= len(query_terms) + 1
        
        scores = []
        for chunk_tf in self.chunk_tfs:
            score = np.dot(query_tf * self.idf, chunk_tf * self.idf)
            scores.append(score)
        return scores
    
    def find_relevant_chunks(self, query: str, top_k: int = 30, semantic_weight: float = 0.7) -> List[Dict]:
        """
        Hybrid search: semantic + keyword
        Returns top chunks after fusion and optional diversity re-ranking
        """
        # Query embedding
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=[query.lower()]  # Normalize
            )
            query_embedding = np.array(response.data[0].embedding)
            query_embedding /= np.linalg.norm(query_embedding)

            # Track cost
            query_tokens = len(self.tokenizer.encode(query))
            self.track_cost(query_tokens, 0, self.embedding_model)
        except Exception as e:
            print(f"Error creating query embedding: {e}")
            return []

        # Semantic similarities
        semantic_sim = []
        for emb in self.embeddings:
            sim = np.dot(query_embedding, emb)
            semantic_sim.append(sim)

        # Keyword scores (TF-IDF)
        query_terms = query.lower().split()
        tfidf_scores = self.compute_tfidf_scores(query_terms)

        # Fuse scores
        fused_scores = []
        for i in range(len(self.chunks)):
            score = semantic_weight * semantic_sim[i] + (1 - semantic_weight) * tfidf_scores[i]
            fused_scores.append((i, score))

        # Sort and get top_k
        fused_scores.sort(key=lambda x: x[1], reverse=True)

        # Diversity re-ranking if enabled
        if self.enable_diversity:
            relevant_chunks = self._diversity_rerank(fused_scores, top_k)
        else:
            top_indices = [idx for idx, _ in fused_scores[:top_k]]
            relevant_chunks = []
            for idx in top_indices:
                chunk = self.chunks[idx].copy()
                chunk['relevance_score'] = dict(fused_scores)[idx]
                relevant_chunks.append(chunk)

        return relevant_chunks

    def _diversity_rerank(self, scored_chunks: List[tuple], top_k: int) -> List[Dict]:
        """
        Re-rank chunks to promote diversity (avoid too many chunks from same post)
        """
        selected = []
        post_counts = defaultdict(int)
        max_per_post = 3  # Maximum chunks from same post

        for idx, score in scored_chunks:
            post_idx = self.chunks[idx]['post_idx']

            # Skip if we already have too many from this post
            if post_counts[post_idx] >= max_per_post:
                continue

            chunk = self.chunks[idx].copy()
            chunk['relevance_score'] = score
            selected.append(chunk)
            post_counts[post_idx] += 1

            if len(selected) >= top_k:
                break

        return selected
    
    def compress_context(self, chunks: List[Dict], query: str) -> str:
        """Compress chunks using OpenAI to summarize for more context (expensive!)"""
        compressed_parts = []
        print(f"[Compression] Compressing {len(chunks)} chunks (this will cost ~${len(chunks) * 0.007:.3f})...")

        for i, chunk in enumerate(chunks):
            prompt = f"Summarize this text concisely, focusing on aspects relevant to: '{query}'\n\nText: {chunk['text'][:2000]}"
            try:
                response = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=150
                )
                summary = response.choices[0].message.content

                # Track cost
                input_tokens = len(self.tokenizer.encode(prompt))
                output_tokens = len(self.tokenizer.encode(summary))
                self.track_cost(input_tokens, output_tokens, self.chat_model)

                compressed_parts.append(summary)
                if (i + 1) % 10 == 0:
                    print(f"[Compression] Processed {i + 1}/{len(chunks)} chunks...")
            except Exception as e:
                print(f"[Warning] Compression failed for chunk {i}: {e}")
                summary = chunk['text'][:500]  # Fallback
                compressed_parts.append(summary)

        return "\n\n".join(compressed_parts)
    
    def build_context(self, chunks: List[Dict], query: str) -> str:
        """Build context from chunks with optional compression"""
        # Group chunks by post_idx to reconstruct
        post_groups = defaultdict(list)
        for chunk in chunks:
            post_groups[chunk['post_idx']].append(chunk)

        context_parts = []
        for post_idx, group in post_groups.items():
            metadata = group[0]['metadata']
            combined_text = " ".join(c['text'] for c in group)  # Recombine for coherence
            context_parts.append(f"Post {post_idx}:")
            context_parts.append(f"Date: {metadata.get('date', 'Unknown')}")
            context_parts.append(f"User: {metadata.get('user', {}).get('first_name', 'Unknown')} (@{metadata.get('user', {}).get('username', 'Unknown')})")
            context_parts.append(f"Content: {combined_text[:1000]}...")  # Initial truncate
            context_parts.append(f"Telegram URL: {metadata.get('telegram_url', 'N/A')}")
            context_parts.append("")

        full_context = "\n".join(context_parts)
        context_tokens = len(self.tokenizer.encode(full_context))

        # Compress only if enabled AND exceeds threshold
        if self.enable_compression and context_tokens > self.compression_threshold:
            print(f"[Context] {context_tokens} tokens exceeds threshold ({self.compression_threshold}), compressing...")
            full_context = self.compress_context(chunks, query)
        else:
            if context_tokens > self.compression_threshold:
                print(f"[Context] {context_tokens} tokens exceeds threshold, but compression is disabled")
            else:
                print(f"[Context] Using {context_tokens} tokens (under threshold: {self.compression_threshold})")

        return full_context
    
    def answer_question(self, question: str, context_posts: int = None, return_sources: bool = False):
        """
        Answer a question about the Telegram posts with improved context

        Args:
            question: The question to answer
            context_posts: Number of context posts to use
            return_sources: If True, return dict with answer and sources. If False, return just answer string.

        Returns:
            If return_sources=False: str (answer text)
            If return_sources=True: dict with keys 'answer' (str) and 'sources' (list of dicts)
        """
        if context_posts is None:
            context_posts = self.default_context_posts

        # Find relevant chunks (larger top_k for hybrid)
        relevant_chunks = self.find_relevant_chunks(question, top_k=context_posts * 2)  # Oversample

        if not relevant_chunks:
            answer = "I couldn't find any relevant posts to answer your question."
            if return_sources:
                return {"answer": answer, "sources": []}
            return answer

        # Build compressed context
        context = self.build_context(relevant_chunks, question)

        # Create the prompt
        system_prompt = """You are a helpful assistant analyzing Telegram posts.
        Based on the provided context, answer the user's question accurately and concisely using as much relevant information as possible.
        If the answer cannot be found in the context, say so.
        When referencing specific posts, ALWAYS include:
        1. The date of the post
        2. The author's name
        3. The Telegram URL link to the post

        Format links as clickable references in your answer."""

        user_prompt = f"""Context (relevant Telegram posts and summaries):
{context}

Question: {question}

Please provide a clear and accurate answer based on the posts above."""

        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000  # Increased for more detailed answers
            )

            # Track cost
            input_tokens = len(self.tokenizer.encode(system_prompt + user_prompt))
            output_tokens = len(self.tokenizer.encode(response.choices[0].message.content))
            self.track_cost(input_tokens, output_tokens, self.chat_model)

            answer = response.choices[0].message.content

            # If sources are requested, extract and format them
            if return_sources:
                sources = self._extract_sources_from_chunks(relevant_chunks)
                return {"answer": answer, "sources": sources}

            return answer
        except Exception as e:
            error_msg = f"Error generating answer: {e}"
            if return_sources:
                return {"answer": error_msg, "sources": []}
            return error_msg

    def _extract_sources_from_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Extract source information from chunks, grouped by post to avoid duplicates
        """
        # Group by post_idx to avoid duplicate sources from same post
        seen_posts = set()
        sources = []

        for chunk in chunks:
            post_idx = chunk['post_idx']
            if post_idx in seen_posts:
                continue
            seen_posts.add(post_idx)

            metadata = chunk['metadata']
            user = metadata.get('user', {})

            source = {
                'date': metadata.get('date', 'Unknown'),
                'author': f"{user.get('first_name', 'Unknown')} (@{user.get('username', 'Unknown')})",
                'url': metadata.get('telegram_url', ''),
                'relevance_score': round(chunk.get('relevance_score', 0), 3)
            }
            sources.append(source)

        # Sort by relevance score
        sources.sort(key=lambda x: x['relevance_score'], reverse=True)

        return sources
    
    # The interactive_chat and show_statistics remain similar, with potential updates for filtering
    def interactive_chat(self):
        """Run an interactive chat session"""
        print("\n" + "="*60)
        print("Kaine AI - Telegram Posts Q&A Tool")
        print("="*60)
        print(f"Loaded {len(self.posts)} posts, {len(self.chunks)} chunks from {self.json_file}")
        print("\nConfiguration:")
        print(f"  - Embedding Model: {self.embedding_model}")
        print(f"  - Chat Model: {self.chat_model}")
        print(f"  - Default Context Posts: {self.default_context_posts}")
        print(f"  - Compression: {'Enabled' if self.enable_compression else 'Disabled'} (threshold: {self.compression_threshold:,} tokens)")
        print(f"  - Diversity Re-ranking: {'Enabled' if self.enable_diversity else 'Disabled'}")
        print(f"  - Cost Tracking: {'Enabled' if self.enable_cost_tracking else 'Disabled'}")
        print("\nYou can ask questions about the posts. Type 'quit' to exit.")
        print("Type 'stats' to see statistics about the posts.")
        print("="*60 + "\n")
        
        while True:
            question = input("\n🤔 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if question.lower() == 'stats':
                self.show_statistics()
                continue
            
            if not question:
                print("Please enter a question.")
                continue
            
            print("\n🔍 Searching for relevant posts (hybrid mode)...")
            answer = self.answer_question(question)
            
            print("\n💡 Answer:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
    
    def show_statistics(self):
        """Show statistics about the loaded posts"""
        print("\n📊 Statistics:")
        print("-" * 40)
        print(f"Total posts: {len(self.posts)}")
        print(f"Total chunks: {len(self.chunks)}")
        
        # Count by user
        users = {}
        for post in self.posts:
            user_id = post.get('user', {}).get('id')
            if user_id:
                users[user_id] = users.get(user_id, 0) + 1
        
        print(f"Unique users: {len(users)}")
        
        # Count by type
        types = {}
        for post in self.posts:
            post_type = post.get('message_type', 'unknown')
            types[post_type] = types.get(post_type, 0) + 1
        
        print("\nPost types:")
        for post_type, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {post_type}: {count}")
        
        # Date range
        dates = [post.get('date') for post in self.posts if post.get('date')]
        if dates:
            print(f"\nDate range: {min(dates)} to {max(dates)}")
        
        # Posts with media
        media_count = sum(1 for post in self.posts if post.get('media'))
        print(f"Posts with media: {media_count}")
        
        print("-" * 40)


def main():
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key not found!")
        print("Please set your API key in the .env file or using:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\nTo use .env file, create a file named '.env' with:")
        print("  OPENAI_API_KEY=your-api-key-here")
        sys.exit(1)
    
    # Check for JSON file argument or use default
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Try data/ directory first with different filenames
        if os.path.exists('data/mrkainez_posts.json'):
            json_file = 'data/mrkainez_posts.json'
        elif os.path.exists('data/sample_posts.json'):
            json_file = 'data/sample_posts.json'
        else:
            json_file = 'sample_posts.json'

    # Check if file exists
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found!")
        print("\nUsage: python kaine_ai.py [json_file]")
        print("Example: python kaine_ai.py data/my_posts.json")
        print("\nMake sure your Telegram JSON data file is in the 'data/' directory.")
        sys.exit(1)
    
    try:
        # Create Q&A tool
        qa_tool = TelegramQA(json_file, api_key)
        
        # Start interactive chat
        qa_tool.interactive_chat()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()