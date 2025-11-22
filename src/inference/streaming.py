from transformers import TextIteratorStreamer
from threading import Thread
import json

def generate_stream(model, tokenizer, prompt, max_tokens=512):
    """
    Generator that yields tokens as they are generated.
    """
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    
    generation_kwargs = dict(
        inputs=tokenizer(prompt, return_tensors="pt"),
        streamer=streamer,
        max_new_tokens=max_tokens
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    for text in streamer:
        yield text
