from prometheus_client import Counter, Histogram, Gauge

# Request metrics
request_count = Counter('llm_requests_total', 'Total requests')
request_latency = Histogram('llm_request_latency_seconds', 'Request latency')
tokens_generated = Counter('llm_tokens_generated_total', 'Total tokens generated')

# Model metrics
model_load_time = Gauge('llm_model_load_seconds', 'Model load time')
gpu_memory_used = Gauge('llm_gpu_memory_bytes', 'GPU memory used')
active_requests = Gauge('llm_active_requests', 'Active requests')

# Cache metrics
cache_hits = Counter('llm_cache_hits_total', 'Cache hits')
cache_misses = Counter('llm_cache_misses_total', 'Cache misses')
