import torch
import torch.quantization
from torch.cuda import memory_allocated
from transformers import AutoTokenizer, AutoModelForCausalLM, QuantoConfig, GPT2LMHeadModel
from optimum.quanto import QuantizedModelForCausalLM, qint4
import time

# @torch.no_grad()
# def golden_greedy_decoding_wo_cache(batch):
#     tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda:1')
#     res = tokenized_batch['input_ids']
#     start_time = time.time()
#     for timestep in range(MAX_NEW_LENGTH):
#         tokenized_batch = original_model.prepare_inputs_for_generation(**tokenized_batch)
#         outputs = original_model(**tokenized_batch)
#         output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
#         tokenized_batch = {
#             "input_ids": torch.cat([tokenized_batch['input_ids'], output_tokens], dim=-1),
#             "attention_mask": torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1),
#         }
#         res = torch.cat([res, output_tokens], dim=-1)
    
#     return res, time.time() - start_time

if __name__ == "__main__":
    MAX_NEW_LENGTH = 100
    bsz = 16

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    original_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map='cuda:1')
    quantized_model = QuantizedModelForCausalLM.quantize(original_model, weights=qint4, exclude="exclude='lm_head'")
    # quantized_model.save_pretrained('./gpt2-int4-quantized')
    # quantized_model = QuantizedModelForCausalLM.from_pretrained('./gpt2-int4-quantized', device_map='cuda:1')
    # quantized_model.to("cuda:1")
    print(quantized_model.transformer.h[0].attn.c_attn.weight)
    # print(memory_allocated("cuda:1"))