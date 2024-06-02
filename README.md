<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Genshin-Impact-Character-Chat</h3>

  <p align="center">
   		Genshin Impact Character Chat Models tuned by Lora on LLM (build by Qwen1.5-7B-Chat and Mistral-7b-Instruct-v0.3)
    <br />
  </p>
</p>

[中文介绍](README.md)

## Brief introduction

### BackGround
[Genshin Impact](https://genshin.hoyoverse.com/en/) is an action role-playing game developed by miHoYo, published by miHoYo in mainland China and worldwide by Cognosphere, 
HoYoverse. The game features an anime-style open-world environment and an action-based battle system using elemental magic and character-switching. 

In the Game, one can play many Characters to explore the amazing open-world environment. <br/>
This project is an attempt to use game characters in daily chat and plot chat.

## Installation and Running Results
This project have three llm inference types: transformers, llama-cpp-python and vLLM. Below are commands to install and run different inference running demos.<br/>
demo index with [3, 5, 7] need run vllm server,before run gradio script. 3 need run one server, [5, 7] need run two servers.

### Install commands
|Index| Chat Type | LLM inference type | Install Command in Linux |
|------|-------|---------|--------|
|1| daily_chatbots | transformers | pip install -r transformer_requirements.txt |
|2| daily_chatbots | llama-cpp-python  | pip install -r transformer_requirements.txt && pip install llama-cpp-python | 
|3|daily_chatbots | vLLM| pip install -r transformer_requirements.txt && pip install vllm | 
| 4|plot_chatbots | transformers | pip install -r transformer_requirements.txt | 
| 5|plot_chatbots | vLLM  | pip install -r transformer_requirements.txt && pip install vllm |
| 6|plot_chatbots | transformers  | pip install -r transformer_requirements.txt | 
|7|plot_chatbots | vLLM  | pip install -r transformer_requirements.txt && pip install vllm |

### Runing commands and GPU requirements
|Index|(If needed, run this before run gradio demo)Run vLLM Server| Run Gradio Demo Command (go to 127.0.0.1:7860) |GPU memory requirements or GPU cards environment|
|------|--------|--------|----|
|1|None| python daily_chatbots/genshin_impact_daily_chatbot_transformer_gradio.py | 3060x1 below 12gb|
|2|None|python daily_chatbots/genshin_impact_daily_chatbot_llama_cpp_gradio.py |3060x1 below 12gb|
|3|python -m vllm.entrypoints.openai.api_server --model svjack/Genshin_Impact_Qwen_1_5_Chat_sharegpt_roleplay_chat_AWQ --dtype auto --api-key token-abc123 --quantization awq --max-model-len 6000 --gpu-memory-utilization 0.9| python daily_chatbots/genshin_impact_daily_chatbot_vllm_gradio.py |3060x1 below 12gb|
|4|None|python plot_chatbots/genshin_impact_plot_chatbot_mistral_v3_transformer_gradio.py | 3060x1 below 12gb|
|5|python -m vllm.entrypoints.openai.api_server --model svjack/Genshin_Impact_Mistral_v3_Plot_Chat_roleplay_chat_AWQ --dtype auto --api-key token-abc123 --quantization awq --max-model-len 4000 --gpu-memory-utilization 0.35 --port 8000
|5|python -m vllm.entrypoints.openai.api_server --model svjack/DPO_Genshin_Impact_Mistral_Plot_Engine_Step_Json_Short_AWQ --dtype auto --api-key token-abc123 --quantization awq --max-model-len 2000 --gpu-memory-utilization 0.35 --port 8001| python plot_chatbots/genshin_impact_plot_chatbot_mistral_v3_vllm_gradio.py | A4000x1 17gb|
|6|None|python plot_chatbots/genshin_impact_plot_chatbot_qwen_7b_transformer_gradio.py | 3060x1 below 12gb|
|7|python -m vllm.entrypoints.openai.api_server --model svjack/Genshin_Impact_Qwen_1_5_Plot_Chat_roleplay_chat_AWQ --dtype auto --api-key token-abc123 --tensor-parallel-size 2 --quantization awq --max-model-len 4000 --gpu-memory-utilization 0.35 --port 8000
|7|python -m vllm.entrypoints.openai.api_server --model svjack/DPO_Genshin_Impact_Qwen_1_5_Plot_Engine_Step_Json_Short_AWQ --dtype auto --api-key token-abc123 --tensor-parallel-size 2 --quantization awq --max-model-len 2000 --gpu-memory-utilization 0.35 --port 8001| python plot_chatbots/genshin_impact_plot_chatbot_qwen_7b_vllm_gradio.py | A4000x2 34gb|

### Running Results
#### daily_chatbots/genshin_impact_daily_chatbot_transformer_gradio.py
##### Screenshot 

##### Video

#### daily_chatbots/genshin_impact_daily_chatbot_llama_cpp_gradio.py

##### Screenshot 

##### Video

#### daily_chatbots/genshin_impact_daily_chatbot_vllm_gradio.py

##### Screenshot 

##### Video

#### plot_chatbots/genshin_impact_plot_chatbot_mistral_v3_transformer_gradio.py

##### Screenshot 

##### Video

#### plot_chatbots/genshin_impact_plot_chatbot_mistral_v3_vllm_gradio.py

##### Screenshot 

##### Video

#### plot_chatbots/genshin_impact_plot_chatbot_qwen_7b_transformer_gradio.py

##### Screenshot 

##### Video

#### plot_chatbots/genshin_impact_plot_chatbot_qwen_7b_vllm_gradio.py

##### Screenshot 

##### Video



