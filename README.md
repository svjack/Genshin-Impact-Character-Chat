<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Genshin-Impact-Character-Chat</h3>

  <p align="center">
   		使用Lora在LLM上微调的原神角色对话模型 (由 Qwen1.5-7B-Chat 和 Mistral-7b-Instruct-v0.3 构建)
    <br />
  </p>
</p>

[In English](README_EN.md)

## 简要引述

### 背景
[Genshin Impact](https://genshin.hoyoverse.com/en/)是miHoYo开发的动作角色扮演游戏，由HoYoverse在大陆中国和全球发布 。该游戏具有动画风格的开放世界环境和基于元素魔法和角色换位的战斗系统。

在游戏中，玩家可以操纵很多个角色来探索壮美的开放世界环境。<br/><br/>
本项目是一次尝试将游戏角色运用到日常聊天和剧情聊天中。 <br/> 在daily_chatbots（日常聊天）的帮助下，您将更深入地了解角色的性格和背景。<br/>在plot_chatbots（剧情聊天）的帮助下，您将更好地了解故事情节以及角色在故事情节中的作用。

### 项目特点

* 1. 这个工程基于Qwen1.5-7B-Chat和Mistral-v3.0训练了两类大模型，包括游戏场景下的日常聊天和剧情聊天两种场景。

* 2. 所有本工程涉及到的模型训练所使用的数据集全部由大模型进行标注生成。

* 3. 日常聊天场景功能由基于角色身份信息的对话能力及角色间日常故事引擎(用于生成基本故事背景和推进故事背景)两部分构成。

* 4. 剧情聊天场景功能由基于角色身份信息的对话能力、全局剧情信息、全局剧情到当前对话剧情到推理引擎（根据全局背景生成二角色对话子背景，可以选取使用全局背景或子背景进行对话，选取全局背景一般是对话者为旅行者和派蒙时，可以快速在更大故事范围体验剧情；选取当前对话剧情一般是旅行者或派蒙和其他NPC对话时，用于了解该NPC在剧情中的局部剧情推进作用）三部分组成。

* 5. 除了使用了常见的SFT微调方法外，对于剧情聊天场景下全局故事背景到当前故事背景到的推理引擎训练也测试了DPO（ORPO）方法，在一定程度上减少重复推理和推理链条顺序颠倒问题。


* 6. 测试了transformers、llama-cpp-python、vLLM三种推理框架在这两个对话场景下的使用，在transformers场景下使用lora switch的方法给出低显存占用条件下的功能实现，在llama-cpp-python、vLLM场景下通过模型分别合并部署和加速给出高推理速度需求下的功能实现，可根据需求灵活选取。

* 7. 提供了webui进行调用。

## 安装和运行结果
该项目具有三种 llm 推理类型：transformers、llama-cpp-python 和 vLLM。以下是安装和运行不同推理运行演示的命令。<br/>
带有[3,5,7]的演示索引需要在运行gradio脚本之前运行vllm服务。 3需要运行一个服务，[5, 7]需要运行两个服务。

### 安装命令
|Index| 聊天类型 | LLM 推理类型 | Linux 环境安装命令 |
|------|-------|---------|--------|
|1| daily_chatbots | transformers | pip install -r transformer_requirements.txt |
|2| daily_chatbots | llama-cpp-python  | pip install -r transformer_requirements.txt && pip install llama-cpp-python | 
|3|daily_chatbots | vLLM| pip install -r transformer_requirements.txt && pip install vllm | 
| 4|plot_chatbots | transformers | pip install -r transformer_requirements.txt | 
| 5|plot_chatbots | vLLM  | pip install -r transformer_requirements.txt && pip install vllm |
| 6|plot_chatbots | transformers  | pip install -r transformer_requirements.txt | 
|7|plot_chatbots | vLLM  | pip install -r transformer_requirements.txt && pip install vllm |

### 运行命令和GPU资源需求
|索引|(在需求时，在运行gradio命令前运行此命令) 运行 vLLM 服务| 运行 Gradio Demo 命令 (查看 127.0.0.1:7860) |GPU 资源需求 或 显卡环境需求|
|------|--------|--------|----|
|1|None| python daily_chatbots/genshin_impact_daily_chatbot_transformer_gradio.py | 3060x1 below 12gb|
|2|None|python daily_chatbots/genshin_impact_daily_chatbot_llama_cpp_gradio.py |3060x1 below 12gb|
|3|python -m vllm.entrypoints.openai.api_server --model svjack/Genshin_Impact_Qwen_1_5_Chat_sharegpt_roleplay_chat_AWQ --dtype auto --api-key token-abc123 --quantization awq --max-model-len 6000 --gpu-memory-utilization 0.9| python daily_chatbots/genshin_impact_daily_chatbot_vllm_gradio.py |A4000x1 17gb|
|4|None|python plot_chatbots/genshin_impact_plot_chatbot_mistral_v3_transformer_gradio.py | 3060x1 below 12gb|
|5|python -m vllm.entrypoints.openai.api_server --model svjack/Genshin_Impact_Mistral_v3_Plot_Chat_roleplay_chat_AWQ --dtype auto --api-key token-abc123 --quantization awq --max-model-len 4000 --gpu-memory-utilization 0.35 --port 8000
|5|python -m vllm.entrypoints.openai.api_server --model svjack/DPO_Genshin_Impact_Mistral_Plot_Engine_Step_Json_Short_AWQ --dtype auto --api-key token-abc123 --quantization awq --max-model-len 2000 --gpu-memory-utilization 0.35 --port 8001| python plot_chatbots/genshin_impact_plot_chatbot_mistral_v3_vllm_gradio.py | A4000x1 17gb|
|6|None|python plot_chatbots/genshin_impact_plot_chatbot_qwen_7b_transformer_gradio.py | 3060x1 below 12gb|
|7|python -m vllm.entrypoints.openai.api_server --model svjack/Genshin_Impact_Qwen_1_5_Plot_Chat_roleplay_chat_AWQ --dtype auto --api-key token-abc123 --tensor-parallel-size 2 --quantization awq --max-model-len 4000 --gpu-memory-utilization 0.35 --port 8000
|7|python -m vllm.entrypoints.openai.api_server --model svjack/DPO_Genshin_Impact_Qwen_1_5_Plot_Engine_Step_Json_Short_AWQ --dtype auto --api-key token-abc123 --tensor-parallel-size 2 --quantization awq --max-model-len 2000 --gpu-memory-utilization 0.35 --port 8001| python plot_chatbots/genshin_impact_plot_chatbot_qwen_7b_vllm_gradio.py | A4000x2 34gb|

### 运行结果
点击下面的图片，在youtube上浏览例子调用视频

#### * 1 daily_chatbots/genshin_impact_daily_chatbot_transformer_gradio.py
[![Genshin Impact Qwen-1.5-7B-Chat Sharegpt Roleplay Turned Transformer Bot](https://img.youtube.com/vi/u8PJWqzhidg/0.jpg)](https://www.youtube.com/watch?v=u8PJWqzhidg) <br/>
 <br/>


#### * 2 daily_chatbots/genshin_impact_daily_chatbot_llama_cpp_gradio.py
[![Genshin Impact Qwen-1.5-7B-Chat Sharegpt Roleplay Turned LLama-CPP Bot](https://img.youtube.com/vi/5duV_UVdhCc/0.jpg)](https://www.youtube.com/watch?v=5duV_UVdhCc) <br/>
 <br/>

#### * 3 daily_chatbots/genshin_impact_daily_chatbot_vllm_gradio.py
[![Genshin Impact Qwen-1.5-7B-Chat Sharegpt Roleplay Turned vLLM Bot](https://img.youtube.com/vi/N1MSLyL3im0/0.jpg)](https://www.youtube.com/watch?v=N1MSLyL3im0) <br/>
 <br/>

#### * 4 plot_chatbots/genshin_impact_plot_chatbot_mistral_v3_transformer_gradio.py
[![Genshin Impact Mistral-7B-instruct-v3 Plot Roleplay Turned Transformer Bot](https://img.youtube.com/vi/G7sW5t0Mhdc/0.jpg)](https://www.youtube.com/watch?v=G7sW5t0Mhdc) <br/>
 <br/>

#### * 5 plot_chatbots/genshin_impact_plot_chatbot_mistral_v3_vllm_gradio.py
[![Genshin Impact Mistral-7B-instruct-v0.3 Plot Roleplay Turned vLLM Bot](https://img.youtube.com/vi/rRPRQRE1zkw/0.jpg)](https://www.youtube.com/watch?v=rRPRQRE1zkw) <br/>
 <br/>

#### * 6 plot_chatbots/genshin_impact_plot_chatbot_qwen_7b_transformer_gradio.py
[![Genshin Impact Qwen-1.5-7B-Chat Plot Roleplay Turned Transformer Bot](https://img.youtube.com/vi/kzQSHdcbg1E/0.jpg)](https://www.youtube.com/watch?v=kzQSHdcbg1E) <br/>
 <br/>

#### * 7 plot_chatbots/genshin_impact_plot_chatbot_qwen_7b_vllm_gradio.py
[![Genshin Impact Qwen-1.5-7B-Chat Plot Roleplay Tuned vLLM Bot](https://img.youtube.com/vi/XgKArKVgZIM/0.jpg)](https://www.youtube.com/watch?v=XgKArKVgZIM) <br/>
 <br/>

#### 注意
* 1 当运行 python plot_chatbots/genshin_impact_plot_chatbot_mistral_v3_transformer_gradio.py 时，您应该先登录您的 Huggingface 帐户以使用基本模型 Mistral-7b-Instruct-v0.3
* 2 daily_chatbots中可以切换角色，daily_chatbots中大约有75个角色，你可以有很多聊天组合，看看https://github.com/svjack/Genshin-Impact-Character-Instruction 查看这些角色。
* 3 在plot_chatbots中，大约有630个可以在情节中进行聊天的角色，请查看https://huggingface.co/datasets/svjack/Genshin-Impact-Plot-Character-Portrait-Merged 查看这些角色。
* 4 在plot_chatbots中，您可以自由选择章节及其相关的故事背景、事件起因、事件进程、事件反转、事件结束、事件意义以及后续情节等。
* 5 在plot_chatbots中，建议使用默认的对话者1，模型是以对话者1的角度训练的。
* 6 在plot_chatbots中，当前对话背景是根据全局对话背景生成的。您可以在右侧页面选择使用全局对话背景或当前对话背景，并根据索引选择背景和对应的​​对话者。
* 7 从性能角度考虑，我建议运行 vLLM 推理演示 ([3, 5, 7])。
* 8 plot_chatbots中使用的有关原神剧情的背景预构建信息（ https://huggingface.co/datasets/svjack/Genshin-Impact-Plot-Summary ）是由AI生成的，因此它们包含一些不准确的错误。<br/>
    但由于 gradio 演示页面中的字段是可编辑的，因此您可以将自己的编辑结果粘贴到字段中或编辑任何您不喜欢的内容。（或者从互联网里面找到一些剧情信息黏贴到里面）<br/>
    此功能也适用于 daily_chatbots。
* 9 与daily_chatbots聊天时，相对灵活自由，与plot_chatbots聊天时，聊天机器人会致力于推进剧情，需要严格按照剧情执行。

## 模型
|索引|聊天类型 | LLM 推理类型|Huggingface 链接|执行的任务|
|---------|--------|--------|-----|----|
|1| daily_chatbots | transformers | https://huggingface.co/svjack/Genshin_Impact_Qwen_1_5_Chat_sharegpt_roleplay_chat_lora_small |日常聊天|
|2| daily_chatbots | llama-cpp-python  | https://huggingface.co/svjack/Genshin_Impact_Qwen_1_5_Chat_sharegpt_roleplay_chat_GGUF |日常聊天 |
|3|daily_chatbots | vLLM| https://huggingface.co/svjack/Genshin_Impact_Qwen_1_5_Chat_sharegpt_roleplay_chat_AWQ | 日常聊天|
| 4|plot_chatbots | transformers | https://huggingface.co/svjack/Genshin_Impact_Mistral_v3_Plot_Chat_roleplay_chat_lora_small | 剧情聊天 |
| 4|plot_chatbots | transformers | https://huggingface.co/svjack/DPO_Genshin_Impact_Mistral_Plot_Engine_Step_Json_Short_lora_small |剧情引擎 |
| 5|plot_chatbots | vLLM  | https://huggingface.co/svjack/Genshin_Impact_Mistral_v3_Plot_Chat_roleplay_chat_AWQ |剧情聊天|
| 5|plot_chatbots | vLLM  | https://huggingface.co/svjack/DPO_Genshin_Impact_Mistral_Plot_Engine_Step_Json_Short_AWQ|剧情引擎|
| 6|plot_chatbots | transformers  | https://huggingface.co/svjack/Genshin_Impact_Qwen_1_5_Plot_Chat_roleplay_chat_lora_small |剧情聊天|
| 6|plot_chatbots | transformers  | https://huggingface.co/svjack/DPO_Genshin_Impact_Qwen_1_5_Plot_Engine_Step_Json_Short_lora_small | 剧情引擎|
|7|plot_chatbots | vLLM  | https://huggingface.co/svjack/Genshin_Impact_Qwen_1_5_Plot_Chat_roleplay_chat_AWQ |剧情聊天|
|7|plot_chatbots | vLLM  | https://huggingface.co/svjack/DPO_Genshin_Impact_Qwen_1_5_Plot_Engine_Step_Json_Short_AWQ |剧情引擎|
#### 注意 
你可以通过查看上面模型的链接，了解其作为一个模型模块使用程序进行调用的方法及其作用和结果

## 进一步阅读
* 1 一个关于 Genshin Impact 角色指令模型的项目，由 Lora 在 [svjack/Genshin-Impact-Character-Instruction](https://github.com/svjack/Genshin-Impact-Character-Instruction) 中发布的 LLM 调整😊
* 2 我还发布了一个LLM支持原神书目问答的项目（由LangChain Haystack ChatGLM Mistral OLlama构建），尝试在不同的LLM支持的RAG系统上构建中文问答。 <br/>
如果您有兴趣，请看一下[svjack/Genshin-Impact-BookQA-LLM](https://github.com/svjack/Genshin-Impact-BookQA-LLM)😊
* 3 上述项目使用的RAG版本[Qwen1.5-14B-Chat](https://huggingface.co/Qwen/Qwen1.5-14B-Chat-GGUF)已发布于
  [svjack/Genshin-Impact-RAG](https://github.com/svjack/Genshin-Impact-RAG)，这是上述项目的知识库版本， 
  你可以在那里检索角色的知识，并且可以角色扮演的方式由角色回答问题。 😊
  
<!-- CONTACT -->
## Contact

<!--
Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com
-->
svjack - https://huggingface.co/svjack - svjackbt@gmail.com - ehangzhou@outlook.com

<!--
Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
-->
Project Link:[https://github.com/svjack/Genshin-Impact-Character-Chat](https://github.com/svjack/Genshin-Impact-Character-Chat)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Genshin Impact](https://genshin.hoyoverse.com/en/)
* [Huggingface](https://huggingface.co)
* [Qwen/Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat)
* [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
* [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
* [vLLM](https://github.com/vllm-project/vllm)
* [svjack/Genshin-Impact-Character-Instruction](https://github.com/svjack/Genshin-Impact-Character-Instruction)
* [svjack/Genshin-Impact-BookQA-LLM](https://github.com/svjack/Genshin-Impact-BookQA-LLM)
* [svjack/Genshin-Impact-RAG](https://github.com/svjack/Genshin-Impact-RAG)

