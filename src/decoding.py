import time
import torch

@torch.no_grad()
def customized_greedy_decoding(model, tokenizer, device, batch_data, desired_length:int):
    tokenized_batch = tokenizer(batch_data, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    past_key_values = None
    res = tokenized_batch['input_ids']
    
    start_time = time.time()
    
    for _ in range(desired_length):
        outputs = model(
            input_ids=tokenized_batch["input_ids"],
            attention_mask=tokenized_batch["attention_mask"],
            past_key_values=past_key_values
        )
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
        past_key_values = outputs["cached_key_values"]
        tokenized_batch = {
            "input_ids": output_tokens,
            "attention_mask": torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1),
        }
        res = torch.cat([res, output_tokens], dim=-1)
    
    time_consumption = time.time() - start_time
        
    return res, time_consumption


@torch.no_grad()
def golden_greedy_decoding_without_cache(model, tokenizer, device, batch_data, desired_length:int):
    tokenized_batch = tokenizer(batch_data, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    res = tokenized_batch['input_ids']
    
    start_time = time.time()
    
    for _ in range(desired_length):
        tokenized_batch = model.prepare_inputs_for_generation(**tokenized_batch)
        outputs = model(**tokenized_batch)
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
        tokenized_batch = {
            "input_ids": torch.cat([tokenized_batch['input_ids'], output_tokens], dim=-1),
            "attention_mask": torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1),
        }
        res = torch.cat([res, output_tokens], dim=-1)
    
    time_consumption = time.time() - start_time
    
    return res, time_consumption

@torch.no_grad()
def golden_greedy_decoding_with_cache(model, tokenizer, device, batch_data, desired_length:int):
    tokenized_batch = tokenizer(batch_data, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    
    start_time = time.time()

    res = model.generate(
        **tokenized_batch,
        max_new_tokens=desired_length,
        min_new_tokens=desired_length,
        use_cache=True,
        eos_token_id=None
    )
    
    time_consumption = time.time() - start_time
    
    return res, time_consumption

@torch.no_grad()
def model_warming(model, tokenizer, device, batch_data):
    tokenized_batch = tokenizer(batch_data, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    tokenized_batch = model.prepare_inputs_for_generation(**tokenized_batch)
    res = model(**tokenized_batch)
    torch.cuda.synchronize(device)