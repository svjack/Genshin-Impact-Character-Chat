
#### Install commands
| Chat Type | LLM inference type | Install Command in Linux |
|-------|---------|--------|
| daily_chatbots | transformers | pip install -r transformer_requirements.txt |
| daily_chatbots | llama-cpp-python  | pip install -r transformer_requirements.txt && pip install llama-cpp-python | 
| daily_chatbots | vLLM| pip install -r transformer_requirements.txt && pip install vllm | 
| plot_chatbots | transformers | pip install -r transformer_requirements.txt | 
| plot_chatbots | vLLM  | pip install -r transformer_requirements.txt && pip install vllm |
| plot_chatbots | transformers  | pip install -r transformer_requirements.txt | 
| plot_chatbots | vLLM  | pip install -r transformer_requirements.txt && pip install vllm |

#### Runing commands and GPU requirements
| Run Gradio Demo Command (go to 127.0.0.1:7860) |GPU memory requirements or GPU cards environment|
|--------|--------|
| python daily_chatbots/genshin_impact_daily_chatbot_transformer_gradio.py | 3060x1 below 12gb|
|python daily_chatbots/genshin_impact_daily_chatbot_llama_cpp_gradio.py |3060x1 below 12gb|
|python -m vllm.entrypoints.openai.api_server --model svjack/Genshin_Impact_Qwen_1_5_Chat_sharegpt_roleplay_chat_AWQ --dtype auto --api-key token-abc123 --quantization awq --max-model-len 6000 --gpu-memory-utilization 0.9 && python daily_chatbots/genshin_impact_daily_chatbot_vllm_gradio.py |3060x1 below 12gb|
|python plot_chatbots/genshin_impact_plot_chatbot_mistral_v3_transformer_gradio.py | 3060x1 below 12gb|
|python -m vllm.entrypoints.openai.api_server --model svjack/Genshin_Impact_Mistral_v3_Plot_Chat_roleplay_chat_AWQ --dtype auto --api-key token-abc123 --quantization awq --max-model-len 4000 --gpu-memory-utilization 0.35 --port 8000 && python -m vllm.entrypoints.openai.api_server --model svjack/DPO_Genshin_Impact_Mistral_Plot_Engine_Step_Json_Short_AWQ --dtype auto --api-key token-abc123 --quantization awq --max-model-len 2000 --gpu-memory-utilization 0.35 --port 8001 && python plot_chatbots/genshin_impact_plot_chatbot_mistral_v3_vllm_gradio.py | A4000x1 17gb|
|python plot_chatbots/genshin_impact_plot_chatbot_qwen_7b_transformer_gradio.py | 3060x1 below 12gb|
| python -m vllm.entrypoints.openai.api_server --model svjack/Genshin_Impact_Qwen_1_5_Plot_Chat_roleplay_chat_AWQ --dtype auto --api-key token-abc123 --tensor-parallel-size 2 --quantization awq --max-model-len 4000 --gpu-memory-utilization 0.35 --port 8000 && python -m vllm.entrypoints.openai.api_server --model svjack/DPO_Genshin_Impact_Qwen_1_5_Plot_Engine_Step_Json_Short_AWQ --dtype auto --api-key token-abc123 --tensor-parallel-size 2 --quantization awq --max-model-len 2000 --gpu-memory-utilization 0.35 --port 8001 && python plot_chatbots/genshin_impact_plot_chatbot_qwen_7b_vllm_gradio.py | A4000x2 34gb|

 

