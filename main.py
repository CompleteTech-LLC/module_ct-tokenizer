# Refactored Token Module adhering to SOLID Principles

import threading
import time
import hashlib
from queue import Queue, Empty
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import unittest

# Configuration Module (config.py)
class Config:
    MAX_BATCH_SIZE = 5
    BATCH_DELAY = 1.0  # in seconds
    MAX_TOKEN_BUDGET = 10000

# Abstract Interfaces

class ITokenBudgetManager(ABC):
    @abstractmethod
    def can_consume_tokens(self, token_count: int) -> bool:
        """Check if the token budget allows consuming the specified number of tokens."""
        pass

    @abstractmethod
    def consume_tokens(self, token_count: int):
        """Consume a specified number of tokens from the budget."""
        pass

class ITokenUsageDatabaseManager(ABC):
    @abstractmethod
    def update_token_usage(self, module: str, tokens_used: int):
        """Update the token usage for a specific module in the database."""
        pass

class IEmbeddingsManager(ABC):
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[Any]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def get_cached_embedding(self, text_id: str) -> Any:
        """Retrieve a cached embedding by its text ID."""
        pass

    @abstractmethod
    def cache_embedding(self, text_id: str, embedding: Any):
        """Cache an embedding with its corresponding text ID."""
        pass

class IToolsManager(ABC):
    @abstractmethod
    def execute(self, command: str, prompts: List[str]) -> List[Dict[str, Any]]:
        """Execute a tool command with the given prompts."""
        pass

# Concrete Implementations

class TokenBudgetManager(ITokenBudgetManager):
    def __init__(self, max_token_budget: int):
        self.max_token_budget = max_token_budget
        self.current_tokens = max_token_budget
        self.lock = threading.Lock()

    def can_consume_tokens(self, token_count: int) -> bool:
        with self.lock:
            return self.current_tokens >= token_count

    def consume_tokens(self, token_count: int):
        with self.lock:
            if self.current_tokens >= token_count:
                self.current_tokens -= token_count
                print(f"Consumed {token_count} tokens. Remaining: {self.current_tokens}")
            else:
                raise ValueError("Insufficient token budget")

class TokenUsageDatabaseManager(ITokenUsageDatabaseManager):
    def __init__(self):
        # Initialize database connection (mocked for demonstration)
        self.token_usage = {}
        self.lock = threading.Lock()

    def update_token_usage(self, module: str, tokens_used: int):
        with self.lock:
            if module in self.token_usage:
                self.token_usage[module] += tokens_used
            else:
                self.token_usage[module] = tokens_used
            print(f"Updated token usage for {module}: {self.token_usage[module]} tokens used.")

class EmbeddingsManager(IEmbeddingsManager):
    def __init__(self):
        self.cache_manager: Dict[str, Any] = {}  # Simple in-memory cache for demonstration
        self.lock = threading.Lock()

    def generate_embeddings(self, texts: List[str]) -> List[Any]:
        # Generate embeddings for the provided texts
        # Placeholder implementation
        print(f"Generating embeddings for texts: {texts}")
        return [hashlib.sha256(text.encode('utf-8')).hexdigest() for text in texts]

    def get_cached_embedding(self, text_id: str) -> Any:
        with self.lock:
            embedding = self.cache_manager.get(text_id, None)
            if embedding:
                print(f"Retrieved cached embedding for ID {text_id}.")
            return embedding

    def cache_embedding(self, text_id: str, embedding: Any):
        with self.lock:
            self.cache_manager[text_id] = embedding
            print(f"Cached embedding for ID {text_id}.")

class ToolsManager(IToolsManager):
    def execute(self, command: str, prompts: List[str]) -> List[Dict[str, Any]]:
        # Execute the given command with prompts
        # Placeholder implementation
        print(f"Executing command '{command}' with prompts: {prompts}")
        return [{"response": f"Processed: {prompt}"} for prompt in prompts]

# Dependency Injection Container

class DependencyContainer:
    def __init__(self):
        self._services = {}
        self._lock = threading.Lock()

    def register(self, interface: ABC, implementation: Any):
        with self._lock:
            self._services[interface] = implementation
            print(f"Registered {implementation.__class__.__name__} for {interface.__name__}.")

    def resolve(self, interface: ABC) -> Any:
        with self._lock:
            implementation = self._services.get(interface)
            if implementation is None:
                raise ValueError(f"No implementation registered for {interface.__name__}")
            print(f"Resolved {implementation.__class__.__name__} for {interface.__name__}.")
            return implementation

# Refactored BatchRequester

class BatchRequester:
    def __init__(self, 
                 budget_manager: ITokenBudgetManager, 
                 usage_db_manager: ITokenUsageDatabaseManager,
                 tools_manager: IToolsManager,
                 max_batch_size: int = Config.MAX_BATCH_SIZE, 
                 batch_delay: float = Config.BATCH_DELAY):
        self.batch_queue = Queue()
        self.max_batch_size = max_batch_size
        self.batch_delay = batch_delay
        self.responses: Dict[str, str] = {}
        self.lock = threading.Lock()
        self.token_budget_manager = budget_manager
        self.token_usage_db_manager = usage_db_manager
        self.tools_manager = tools_manager
        self.stop_event = threading.Event()
        self.processing_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.processing_thread.start()
        print("BatchRequester initialized and processing thread started.")

    def _process_batches(self):
        while not self.stop_event.is_set():
            batch = []
            prompt_ids = []
            prompts = []
            try:
                prompt_id, prompt = self.batch_queue.get(timeout=self.batch_delay)
                batch.append((prompt_id, prompt))
                prompt_ids.append(prompt_id)
                prompts.append(prompt)
                while len(batch) < self.max_batch_size:
                    prompt_id, prompt = self.batch_queue.get_nowait()
                    batch.append((prompt_id, prompt))
                    prompt_ids.append(prompt_id)
                    prompts.append(prompt)
            except Empty:
                pass
            if batch:
                total_tokens = sum(len(prompt.split()) for prompt in prompts)
                print(f"Processing batch of {len(batch)} prompts with total tokens: {total_tokens}")
                if self.token_budget_manager.can_consume_tokens(total_tokens):
                    try:
                        responses = self.tools_manager.execute("call_llm_batch", prompts=prompts)
                        tokens_used = total_tokens + sum(len(resp.get("response", "").split()) for resp in responses)
                        self.token_usage_db_manager.update_token_usage('batch_requester', tokens_used)
                        self.token_budget_manager.consume_tokens(tokens_used)
                        with self.lock:
                            for prompt_id, response in zip(prompt_ids, responses):
                                self.responses[prompt_id] = response.get("response", "")
                                print(f"Response for ID {prompt_id} stored.")
                    except Exception as e:
                        # Handle exceptions such as tool execution failure
                        with self.lock:
                            for prompt_id in prompt_ids:
                                self.responses[prompt_id] = f"Error: {str(e)}"
                                print(f"Error processing prompt ID {prompt_id}: {str(e)}")
                else:
                    with self.lock:
                        for prompt_id in prompt_ids:
                            self.responses[prompt_id] = "Error: Token budget exceeded"
                            print(f"Token budget exceeded for prompt ID {prompt_id}.")

    def request(self, prompt: str) -> str:
        prompt_id = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        with self.lock:
            if prompt_id in self.responses:
                print(f"Returning cached response for prompt ID {prompt_id}.")
                return self.responses.pop(prompt_id)
        self.batch_queue.put((prompt_id, prompt))
        print(f"Prompt ID {prompt_id} added to the queue.")
        while True:
            with self.lock:
                if prompt_id in self.responses:
                    print(f"Returning response for prompt ID {prompt_id}.")
                    return self.responses.pop(prompt_id)
            time.sleep(0.01)

    def stop(self):
        self.stop_event.set()
        self.processing_thread.join()
        print("BatchRequester processing thread stopped.")

# Refactored BatchEmbeddings

class BatchEmbeddings:
    def __init__(self, 
                 embeddings_manager: IEmbeddingsManager, 
                 max_batch_size: int = 10, 
                 batch_delay: float = 0.5):
        self.embeddings_queue = Queue()
        self.max_batch_size = max_batch_size
        self.batch_delay = batch_delay
        self.embeddings_manager = embeddings_manager
        self.embeddings_results: Dict[str, Any] = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.processing_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.processing_thread.start()
        print("BatchEmbeddings initialized and processing thread started.")

    def _process_batches(self):
        while not self.stop_event.is_set():
            batch = []
            text_ids = []
            texts = []
            try:
                text_id, text = self.embeddings_queue.get(timeout=self.batch_delay)
                batch.append((text_id, text))
                text_ids.append(text_id)
                texts.append(text)
                while len(batch) < self.max_batch_size:
                    text_id, text = self.embeddings_queue.get_nowait()
                    batch.append((text_id, text))
                    text_ids.append(text_id)
                    texts.append(text)
            except Empty:
                pass
            if batch:
                print(f"Processing batch of {len(batch)} texts for embeddings.")
                if texts:
                    try:
                        embeddings = self.embeddings_manager.generate_embeddings(texts)
                        with self.lock:
                            for text_id, embedding in zip(text_ids, embeddings):
                                self.embeddings_results[text_id] = embedding
                                self.embeddings_manager.cache_embedding(text_id, embedding)
                                print(f"Embedding for text ID {text_id} stored and cached.")
                    except Exception as e:
                        # Handle embedding generation failures
                        with self.lock:
                            for text_id in text_ids:
                                self.embeddings_results[text_id] = f"Error: {str(e)}"
                                print(f"Error generating embedding for text ID {text_id}: {str(e)}")

    def request_embedding(self, text: str) -> Any:
        text_id = hashlib.sha256(text.encode('utf-8')).hexdigest()
        with self.lock:
            if text_id in self.embeddings_results:
                print(f"Returning cached embedding for text ID {text_id}.")
                return self.embeddings_results.pop(text_id)
        cached_embedding = self.embeddings_manager.get_cached_embedding(text_id)
        if cached_embedding:
            return cached_embedding
        self.embeddings_queue.put((text_id, text))
        print(f"Text ID {text_id} added to the embeddings queue.")
        while True:
            with self.lock:
                if text_id in self.embeddings_results:
                    print(f"Returning embedding for text ID {text_id}.")
                    return self.embeddings_results.pop(text_id)
            time.sleep(0.01)

    def stop(self):
        self.stop_event.set()
        self.processing_thread.join()
        print("BatchEmbeddings processing thread stopped.")

# Unit Tests

class TestTokenModule(unittest.TestCase):
    def setUp(self):
        # Initialize managers
        self.budget_manager = TokenBudgetManager(max_token_budget=100)
        self.usage_db_manager = TokenUsageDatabaseManager()
        self.tools_manager = ToolsManager()
        self.embeddings_manager = EmbeddingsManager()
        
        # Initialize BatchRequester and BatchEmbeddings with dependency injection
        self.batch_requester = BatchRequester(
            budget_manager=self.budget_manager,
            usage_db_manager=self.usage_db_manager,
            tools_manager=self.tools_manager,
            max_batch_size=2,
            batch_delay=0.1
        )
        
        self.batch_embeddings = BatchEmbeddings(
            embeddings_manager=self.embeddings_manager,
            max_batch_size=2,
            batch_delay=0.1
        )

    def tearDown(self):
        # Ensure threads are properly stopped after tests
        self.batch_requester.stop()
        self.batch_embeddings.stop()

    def test_batch_requester_success(self):
        prompt = "Hello, how are you?"
        response = self.batch_requester.request(prompt)
        self.assertEqual(response, "Processed: Hello, how are you?")

    def test_batch_requester_token_budget_exceeded(self):
        # Consume tokens to exceed the budget
        self.budget_manager.consume_tokens(100)
        prompt = "This should fail due to budget."
        response = self.batch_requester.request(prompt)
        self.assertEqual(response, "Error: Token budget exceeded")

    def test_batch_embeddings_success(self):
        text = "Sample text for embedding."
        embedding = self.batch_embeddings.request_embedding(text)
        expected_embedding = hashlib.sha256(text.encode('utf-8')).hexdigest()
        self.assertEqual(embedding, expected_embedding)

    def test_batch_embeddings_cache(self):
        text = "Cached embedding text."
        # First request to generate and cache
        embedding1 = self.batch_embeddings.request_embedding(text)
        # Second request should retrieve from cache
        embedding2 = self.batch_embeddings.request_embedding(text)
        self.assertEqual(embedding1, embedding2)

    def test_token_usage_update(self):
        prompt = "Track token usage."
        _ = self.batch_requester.request(prompt)
        self.assertIn('batch_requester', self.usage_db_manager.token_usage)
        tokens_used = len(prompt.split()) + len("Processed: " + prompt.split()[-1].split())  # Approximation
        self.assertGreaterEqual(self.usage_db_manager.token_usage['batch_requester'], tokens_used)

# Example Usage

def main():
    # Initialize Dependency Container
    container = DependencyContainer()
    
    # Register concrete implementations
    container.register(ITokenBudgetManager, TokenBudgetManager(max_token_budget=10000))
    container.register(ITokenUsageDatabaseManager, TokenUsageDatabaseManager())
    container.register(IEmbeddingsManager, EmbeddingsManager())
    container.register(IToolsManager, ToolsManager())
    
    # Resolve dependencies
    budget_manager = container.resolve(ITokenBudgetManager)
    usage_db_manager = container.resolve(ITokenUsageDatabaseManager)
    embeddings_manager = container.resolve(IEmbeddingsManager)
    tools_manager_instance = container.resolve(IToolsManager)
    
    # Initialize BatchRequester and BatchEmbeddings with dependency injection
    batch_requester = BatchRequester(
        budget_manager=budget_manager,
        usage_db_manager=usage_db_manager,
        tools_manager=tools_manager_instance,
        max_batch_size=Config.MAX_BATCH_SIZE,
        batch_delay=Config.BATCH_DELAY
    )

    batch_embeddings = BatchEmbeddings(
        embeddings_manager=embeddings_manager,
        max_batch_size=10,
        batch_delay=0.5
    )

    # Example requests
    try:
        response = batch_requester.request("Hello, how are you?")
        print(f"BatchRequester Response: {response}")

        embedding = batch_embeddings.request_embedding("Sample text for embedding.")
        print(f"BatchEmbeddings Result: {embedding}")
    finally:
        # Ensure threads are properly stopped
        batch_requester.stop()
        batch_embeddings.stop()

if __name__ == "__main__":
    # Run example usage
    main()
    
    # Run unit tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
