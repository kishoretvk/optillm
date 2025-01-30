"""
thinkdeeper_plugin.py - Plugin for enhanced thinking capabilities in optillm
"""

import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from typing import Tuple, Dict, Any, List
import logging

# Plugin identifier
SLUG = "thinkdeeper"

# Default configurations
DEFAULT_CONFIG = {
    "replacements": ["\nWait, but", "\nHmm", "\nSo"],
    "min_thinking_tokens": 128,
    "prefill": "",
    "start_think_token": "<think>",
    "end_think_token": "</think>"
}

logger = logging.getLogger(__name__)

class ThinkDeeperProcessor:
    def __init__(self, config: Dict[str, Any], tokenizer, model):
        self.config = {**DEFAULT_CONFIG, **config}
        self.tokenizer = tokenizer
        self.model = model
        
        # Get the actual token IDs for think markers
        tokens = self.tokenizer.encode(f"{self.config['start_think_token']}{self.config['end_think_token']}")
        self._start_think_token = tokens[1]  # Start token is second token
        self.end_think_token = tokens[2]     # End token is third token
        logger.debug(f"Think token IDs - Start: {self._start_think_token} ({self.tokenizer.decode([self._start_think_token])}), End: {self.end_think_token} ({self.tokenizer.decode([self.end_think_token])})")
        

    @torch.inference_mode()
    def reasoning_effort(self, question: str) -> str:
        """
        Generate an enhanced thinking response with extended reasoning.
        
        Args:
            question: The input question to process
            
        Returns:
            The generated response with enhanced thinking
        """
        logger.debug(f"Starting generation for question: {question}")
        tokens = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"{self.config['start_think_token']}\n{self.config['prefill']}"},
            ],
            continue_final_message=True,
            return_tensors="pt"
        )
        tokens = tokens.to(self.model.device)
        # First get the template up to assistant's message
        template_prefix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            continue_final_message=False,
            return_tensors="pt"
        )
        # Get the length of just the template and user message
        template_prefix_text = self.tokenizer.decode(template_prefix[0])
        prefix_len = len(template_prefix_text)

        kv = DynamicCache()
        n_thinking_tokens = 0
        seen_end_think = False
        response_chunks = []
        
        while True:
            out = self.model(input_ids=tokens, past_key_values=kv, use_cache=True)
            next_token = torch.multinomial(
                torch.softmax(out.logits[0, -1, :], dim=-1), 1
            ).item()
            kv = out.past_key_values
            
            next_str = self.tokenizer.decode([next_token])
            logger.debug(f"Generated token {next_token} -> '{next_str}'")

            # Track if we've seen the end think token
            if next_token == self.end_think_token:
                seen_end_think = True
                logger.debug("Found end think token")

            # Need to continue generating if:
            # 1. We hit end think/eos before min tokens OR
            # 2. We hit eos without seeing end think token
            if ((next_token in (self.end_think_token, self.model.config.eos_token_id) 
                 and n_thinking_tokens < self.config["min_thinking_tokens"]) 
                or (next_token == self.model.config.eos_token_id and not seen_end_think)):
                
                replacement = random.choice(self.config["replacements"])
                logger.debug(f"Inserting replacement: '{replacement}' (tokens: {n_thinking_tokens}, seen_end_think: {seen_end_think})")
                response_chunks.append(replacement)
                replacement_tokens = self.tokenizer.encode(replacement)
                n_thinking_tokens += len(replacement_tokens)
                tokens = torch.tensor([replacement_tokens]).to(tokens.device)
                
            elif next_token == self.model.config.eos_token_id and seen_end_think:
                logger.debug("Reached EOS after end think token - stopping generation")
                break
                
            else:
                response_chunks.append(next_str)
                n_thinking_tokens += 1
                tokens = torch.tensor([[next_token]]).to(tokens.device)
                logger.debug(f"Added token to response. Total thinking tokens: {n_thinking_tokens}")

        # Join all chunks and trim off the initial prompt
        full_response = "".join(response_chunks)
        final_response = full_response[prefix_len:]
        
        logger.debug(f"Final response length: {len(final_response)} chars")
        return final_response

def run(system_prompt: str, initial_query: str, client, model: str, request_config: Dict[str, Any] = None) -> Tuple[str, int]:
    """
    Main plugin execution function.
    
    Args:
        system_prompt: System prompt text
        initial_query: Query to process
        client: OpenAI client instance
        model: Model identifier
        request_config: Additional configuration from the request
        
    Returns:
        Tuple of (generated response, completion tokens)
    """
    logger.info("Starting ThinkDeeper processing")
    
    # Extract config from request_config if provided
    config = DEFAULT_CONFIG.copy()
    if request_config:
        thinkdeeper_config = request_config.get("thinkdeeper_config", {})
        # Update only valid keys
        for key in DEFAULT_CONFIG:
            if key in thinkdeeper_config:
                config[key] = thinkdeeper_config[key]
    
    try:      
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model)
        llm = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
        logger.info("Model and tokenizer loaded successfully")
        
        # Create processor and generate response
        processor = ThinkDeeperProcessor(config, tokenizer, llm)
        response = processor.reasoning_effort(initial_query)
        
        # Calculate actual completion tokens
        completion_tokens = len(tokenizer.encode(response))
        logger.info(f"Generation complete. Used {completion_tokens} completion tokens")
        
        return response, completion_tokens
        
    except Exception as e:
        logger.error(f"Error in ThinkDeeper processing: {str(e)}")
        # Fallback to standard response
        return f"Error in enhanced thinking process: {str(e)}", 0