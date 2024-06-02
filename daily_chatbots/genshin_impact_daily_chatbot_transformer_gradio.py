'''
gradio==4.29.0
peft==0.11.1
transformers==4.41.0
bitsandbytes
huggingface_hub
datasets
Pillow==10.3.0
'''

import numpy as np
import pandas as pd
import gradio
import json
import os
import re

from PIL import Image
from threading import Thread

from transformers import TextStreamer, TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from huggingface_hub import snapshot_download

from typing import List, Optional, Tuple, Dict
History = List[Tuple[str, str]]
Messages = List[Dict[str, str]]

import gradio as gr

if not os.path.exists("genshin-impact-character"):
    path = snapshot_download(
        repo_id="svjack/genshin-impact-character",
        repo_type="dataset",
        local_dir="genshin-impact-character",
        local_dir_use_symlinks = False
    )

info_df = pd.read_csv("genshin-impact-character/genshin_impact_background_settings_constrained.csv")
info_df["info"] = info_df["info"].map(eval)

with open("genshin-impact-character/genshin_impact_character_setting.json", "r") as f:
    character_setting_total_dict = json.load(f)

req_dict = {}
for k, v_dict in character_setting_total_dict.items():
    req_dict[k] = {}
    for kk, vv in v_dict.items():
        if kk != "元素力":
            req_dict[k][kk] = vv
character_setting_total_dict = req_dict

def get_character_background_list(info_dict):
    text = []
    if "角色详细" in info_dict["描述"]:
        text.append(info_dict["描述"]["角色详细"])
    if "更多描述" in info_dict["描述"]:
        text.append(info_dict["描述"]["更多描述"])
    return list(map(lambda x: x.replace(" ", "").replace("\n\n", "\n"), text))
def get_character_background(info_dict, all = False):
    if all:
        return "\n".join(get_character_background_list(info_dict))
    else:
        return get_character_background_list(info_dict)[0] if get_character_background_list(info_dict) else ""

pd.DataFrame(
pd.Series(character_setting_total_dict.values()).map(
    lambda x: {
        "性别": x['性别'],
        "国籍": x["国籍"]
    }
).values.tolist()).apply(lambda x: set(x), axis = 0).to_dict()


character_setting_total_dist_dict = {
 '姓名': "",
 '性别': {'少女女性', '少年男性', '成年女性', '成年男性'},
 '国籍': {'枫丹', '璃月', '稻妻', '至冬', '蒙德', '须弥'},
 '身份': "",
 '性格特征': "",
 '角色介绍': "",
 }

def get_character_setting_total_dict(name):
    from copy import deepcopy
    req = deepcopy(character_setting_total_dist_dict)
    if name in character_setting_total_dict:
        for k, v in character_setting_total_dict[name].items():
            req[k] = v
        info_dict = dict(info_df[["title", "info"]].values.tolist())[name]
        req["角色介绍"] = get_character_background(info_dict)
    req["姓名"] = name
    return req

prompt_format_dict = {
    "Basic_Info": ["性别", "国籍", "身份", "性格特征"],

    "两人同属{}": ["国籍"],
    "{}来自{},{}来自{}。": ["姓名", "国籍", "姓名", "国籍"],

    "下面是{}的一些基本信息\n{}": ["姓名", "Basic_Info"],
    "下面是{}的一些基本信息\n{}\n这些是一段角色介绍\n{}": ["姓名", "Basic_Info", "角色介绍"],

    "续写下面的角色介绍，下面是角色介绍的开头。{}是{}。{}": ["姓名", "身份", "Text"],
    "续写下面的角色故事，下面是角色故事的开头。{}是{}。{}": ["姓名", "身份", "Text"],
    "续写下面获得神之眼的过程，下面是开头。{}是{}。{}": ["姓名", "身份", "Text"],
    "{}给你写了一封信，信主题是{}，信的内容是这样的。": ["姓名", "Text"],

    "{}在进行有关{}的聊天时会说什么？": ["姓名", "Text"],
    "{}在{}的时候会说什么？": ["姓名", "Text"],
    "{}在{}时会说什么？": ["姓名", "Text"],
    "关于{}，{}会说什么?": ["Text", "姓名"],
    "当你想要了解{}时": ["姓名"],

    "关于{}，{}会说什么?": ["姓名", "姓名"],
    "从{}那里，可以获得哪些关于{}的信息？": ["姓名", "姓名"]
}

def single_character_prompt_func(name,
    used_prompt_format_dict,
    character_setting_rewrite_dict = {},
    Text = "",
    ):
    assert type(used_prompt_format_dict) == type({})
    assert type(character_setting_rewrite_dict) == type({})
    character_setting_total_dict = get_character_setting_total_dict(name)
    for k, v in character_setting_rewrite_dict.items():
        if k in character_setting_total_dict:
            character_setting_total_dict[k] = v
    key = list(used_prompt_format_dict.keys())[0]
    assert key in prompt_format_dict
    if key == "Basic_Info":
        return "\n".join(
        map(lambda k: "{}:{}".format(k, character_setting_total_dict[k]), prompt_format_dict[key])
        )
    elif key == "两人同属{}":
        return "两人同属{}".format(character_setting_total_dict["国籍"])
    elif key == "下面是{}的一些基本信息\n{}":
        return "下面是{}的一些基本信息\n{}".format(name,
            single_character_prompt_func(name,
                {
                    "Basic_Info": ["性别", "国籍", "身份", "性格特征"]
                },
                character_setting_rewrite_dict
            )
        )
    elif key == "下面是{}的一些基本信息\n{}\n这些是一段角色介绍\n{}":
        return "下面是{}的一些基本信息\n{}\n这些是一段角色介绍\n{}".format(
            name,
            single_character_prompt_func(name,
                {
                    "Basic_Info": ["性别", "国籍", "身份", "性格特征"]
                },
                character_setting_rewrite_dict
            ),
            character_setting_total_dict["角色介绍"]
        )
    elif key == "续写下面的角色介绍，下面是角色介绍的开头。{}是{}。{}":
        return "续写下面的角色介绍，下面是角色介绍的开头。{}是{}。{}".format(
            name,
            character_setting_total_dict["身份"],
            Text
        )
    elif key == "续写下面的角色故事，下面是角色故事的开头。{}是{}。{}":
        return "续写下面的角色故事，下面是角色介绍的开头。{}是{}。{}".format(
            name,
            character_setting_total_dict["身份"],
            Text
        )
    elif key == "续写下面获得神之眼的过程，下面是开头。{}是{}。{}":
        return "续写下面获得神之眼的过程，下面是开头。{}是{}。{}".format(
            name,
            character_setting_total_dict["身份"],
            Text
        )
    elif key == "{}给你写了一封信，信主题是{}，信的内容是这样的。":
        return "{}给你写了一封信，信主题是{}，信的内容是这样的。".format(
            name,
            Text
        )
    elif key == "{}在进行有关{}的聊天时会说什么？":
        return "{}在进行有关{}的聊天时会说什么？".format(
            name,
            Text
        )
    elif key == "{}在{}的时候会说什么？":
        return "{}在{}的时候会说什么？".format(
            name,
            Text
        )
    elif key == "{}在{}时会说什么？":
        return "{}在{}时会说什么？".format(
            name,
            Text
        )
    elif key == "关于{}，{}会说什么?":
        return "关于{}，{}会说什么?".format(
            Text,
            name,
        )
    elif key == "当你想要了解{}时":
        return "当你想要了解{}时".format(
            name,
        )
    return 1 / 0

def two_character_prompt_func(
    name_1,
    name_2,
    used_prompt_format_dict,
    character_setting_rewrite_dict_1 = {},
    character_setting_rewrite_dict_2 = {},
    ):
    assert type(character_setting_rewrite_dict_1) == type({})
    character_setting_total_dict_1 = get_character_setting_total_dict(name_1)
    for k, v in character_setting_rewrite_dict_1.items():
        if k in character_setting_total_dict_1:
            character_setting_total_dict_1[k] = v
    character_setting_total_dict_2 = get_character_setting_total_dict(name_2)
    for k, v in character_setting_rewrite_dict_2.items():
        if k in character_setting_total_dict_2:
            character_setting_total_dict_2[k] = v
    key = list(used_prompt_format_dict.keys())[0]
    assert key in prompt_format_dict
    if key == "关于{}，{}会说什么?":
        return "关于{}，{}会说什么?".format(name_1, name_2)
    elif key == "从{}那里，可以获得哪些关于{}的信息？":
        return "从{}那里，可以获得哪些关于{}的信息？".format(name_1, name_2)
    elif key == "{}来自{},{}来自{}。":
        return "{}来自{},{}来自{}。".format(name_1, character_setting_total_dict_1["国籍"],
        name_2, character_setting_total_dict_2["国籍"],
        )
    return 1 / 0

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat",)
qw_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B-Chat", load_in_4bit = True)
qw_model = PeftModel.from_pretrained(qw_model,
                "svjack/Genshin_Impact_Qwen_1_5_Chat_sharegpt_roleplay_chat_lora_small"
        )
qw_model = qw_model.eval()
streamer = TextStreamer(tokenizer)
iter_streamer = TextIteratorStreamer(tokenizer)

def qwen_hf_predict(messages, qw_model = qw_model,
    tokenizer = tokenizer, streamer = streamer,
    do_sample = True,
    top_p = 0.95,
    top_k = 40,
    max_new_tokens = 2070,
    max_input_length = 3500,
    temperature = 0.9,
    repetition_penalty = 1.0,
    device = "cuda"):

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt",
        add_generation_prompt=True
    )
    model_inputs = encodeds.to(device)

    generated_ids = qw_model.generate(model_inputs, max_new_tokens=max_new_tokens,
                                do_sample=do_sample,
                                  streamer = streamer,
                                  top_p = top_p,
                                  top_k = top_k,
                                  temperature = temperature,
                                  repetition_penalty = repetition_penalty,
                                  )
    out = tokenizer.batch_decode(generated_ids)[0].split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
    return out

#### 1 - Answer questions about Genshin Impact Character using Third Person.
def third_person_instruction(single_name, question):
    assert type(single_name) == type("")
    assert type(question) == type("")
    info_prompt = single_character_prompt_func(
        single_name,
        {
        "下面是{}的一些基本信息\n{}\n这些是一段角色介绍\n{}": ["姓名", "Basic_Info", "角色介绍"]
        },
    )
    messages=[
                    {
                        "role": "system",
                        "content": "",
                    },
                    {
                        "role": "user",
                        "content": '''
                        {info_prompt}
                        {question}
                        '''.format(**{
                            "info_prompt": info_prompt,
                            "question": question
                        })
                    }
    ]
    return messages

#### 2 - Answer questions about Genshin Impact Character using the First Person.
def first_person_instruction(single_name, question):
    assert type(single_name) == type("")
    assert type(question) == type("")
    info_prompt = single_character_prompt_func(
        single_name,
        {
        "下面是{}的一些基本信息\n{}\n这些是一段角色介绍\n{}": ["姓名", "Basic_Info", "角色介绍"]
        },
    )
    messages=[
                    {
                        "role": "system",
                        "content": '''
                        人物设定:
                        {info_prompt}

                        你扮演:{single_name}
                        '''.format(**{
                            "info_prompt": info_prompt,
                            "single_name": single_name
                        })
                    },
                    {
                        "role": "user",
                        "content": question
                    }
    ]
    return messages

##### 4 - Give Story Background between Genshin Impact Characters
def init_background_instruction(single_name_1, single_name_2):
    assert type(single_name_1) == type("")
    assert type(single_name_2) == type("")
    info_prompt_1 = single_character_prompt_func(
        single_name_1,
        {
        "下面是{}的一些基本信息\n{}\n这些是一段角色介绍\n{}": ["姓名", "Basic_Info", "角色介绍"]
        },
    )
    info_prompt_2 = single_character_prompt_func(
        single_name_2,
        {
        "下面是{}的一些基本信息\n{}\n这些是一段角色介绍\n{}": ["姓名", "Basic_Info", "角色介绍"]
        },
    )
    character_setting_total_dict_1 = get_character_setting_total_dict(single_name_1)
    character_setting_total_dict_2 = get_character_setting_total_dict(single_name_2)
    country_prompt = ""
    same_country = character_setting_total_dict_1["国籍"] == character_setting_total_dict_2["国籍"]
    if same_country:
        country_prompt = single_character_prompt_func(
            single_name_1,
            {
                "两人同属{}": ["国籍"]
            },
            )
    else:
        country_prompt = two_character_prompt_func(
                single_name_1,
                single_name_2,
                {
                "{}来自{},{}来自{}。": ["姓名", "国籍", "姓名", "国籍"]
                },
            )
    messages=[
                    {
                            "role": "system",
                            "content": "",
                    },
                    {
                        "role": "user",
                        "content": '''
                        人物设定:
                        {info_prompt_1}
                        {info_prompt_2}
                        {country_prompt}

                        根据上面的人物设定生成发生在{single_name_1}和{single_name_2}之间的故事背景
                        '''.format(**{
                            "info_prompt_1": info_prompt_1,
                            "info_prompt_2": info_prompt_2,
                            "country_prompt": country_prompt,
                            "single_name_1": single_name_1,
                            "single_name_2": single_name_2
                        })
                    }
    ]
    return messages


#### 5 - Role Play Chat with Character Agent, provide your own character setting as another Character At same time, provide above Story Background, as Conversation background.
def init_chat_system_messages(human_name, gpt_name, background):
    assert type(human_name) == type("")
    assert type(gpt_name) == type("")
    assert type(background) == type("")
    info_prompt_1 = single_character_prompt_func(
        human_name,
        {
        "下面是{}的一些基本信息\n{}\n这些是一段角色介绍\n{}": ["姓名", "Basic_Info", "角色介绍"]
        },
    )
    info_prompt_2 = single_character_prompt_func(
        gpt_name,
        {
        "下面是{}的一些基本信息\n{}\n这些是一段角色介绍\n{}": ["姓名", "Basic_Info", "角色介绍"]
        },
    )
    character_setting_total_dict_1 = get_character_setting_total_dict(human_name)
    character_setting_total_dict_2 = get_character_setting_total_dict(gpt_name)
    country_prompt = ""
    same_country = character_setting_total_dict_1["国籍"] == character_setting_total_dict_2["国籍"]
    if same_country:
        country_prompt = single_character_prompt_func(
            human_name,
            {
                "两人同属{}": ["国籍"]
            },
            )
    else:
        country_prompt = two_character_prompt_func(
                human_name,
                gpt_name,
                {
                "{}来自{},{}来自{}。": ["姓名", "国籍", "姓名", "国籍"]
                },
            )
    messages=[
                    {
                        "role": "system",
                        "content": '''
                        人物设定:
                        {info_prompt_1}
                        {info_prompt_2}
                        {country_prompt}

                        背景设定:
                        {background}

                        你扮演:{gpt_name}
                        '''.format(**{
                            "info_prompt_1": info_prompt_1,
                            "info_prompt_2": info_prompt_2,
                            "country_prompt": country_prompt,
                            "background": background,
                            "gpt_name": gpt_name
                        })
                    }
    ]
    return messages

#### 6 - Generate New Story Background, on previous Background and Chat Context.
def new_background_instruction(human_name, gpt_name, previous_background,
    hist_chat_messages
):
    assert type(human_name) == type("")
    assert type(gpt_name) == type("")
    assert type(previous_background) == type("")
    assert type(hist_chat_messages) == type([])
    info_prompt_1 = single_character_prompt_func(
        human_name,
        {
        "下面是{}的一些基本信息\n{}\n这些是一段角色介绍\n{}": ["姓名", "Basic_Info", "角色介绍"]
        },
    )
    info_prompt_2 = single_character_prompt_func(
        gpt_name,
        {
        "下面是{}的一些基本信息\n{}\n这些是一段角色介绍\n{}": ["姓名", "Basic_Info", "角色介绍"]
        },
    )
    character_setting_total_dict_1 = get_character_setting_total_dict(human_name)
    character_setting_total_dict_2 = get_character_setting_total_dict(gpt_name)
    country_prompt = ""
    same_country = character_setting_total_dict_1["国籍"] == character_setting_total_dict_2["国籍"]
    if same_country:
        country_prompt = single_character_prompt_func(
            human_name,
            {
                "两人同属{}": ["国籍"]
            },
            )
    else:
        country_prompt = two_character_prompt_func(
                human_name,
                gpt_name,
                {
                "{}来自{},{}来自{}。": ["姓名", "国籍", "姓名", "国籍"]
                },
            )
    hist_chat_messages_ = list(
        filter(lambda d: d["role"] in ["user", "assistant"], hist_chat_messages)
    )
    chat_context = "\n".join(map(lambda d: "{}:{}".format(
    human_name if d["role"] == "user" else gpt_name ,d["content"]
    ), hist_chat_messages_))
    messages=[
                    {
                        "role": "system",
                        "content": "",
                    },
                    {
                        "role": "user",
                        "content": '''
                        人物设定:
                        {info_prompt_1}
                        {info_prompt_2}
                        {country_prompt}

                        下面是发生在{human_name}和{gpt_name}之间的故事背景:
                        {previous_background}

                        二人发生了如下对话:
                        {chat_context}

                        同时，为推动对话情节发展，请你用类似上面故事背景的风格，给出一个基于上面设定的新故事背景，要求新故事背景与原故事背景有因果联系。
                        使得{human_name}和{gpt_name}可以在新的故事背景中进行互动。
                        要求只输出一行文字，新故事背景中必须提到{human_name}和{gpt_name}。
                        '''.format(**{
                            "info_prompt_1": info_prompt_1,
                            "info_prompt_2": info_prompt_2,
                            "country_prompt": country_prompt,
                            "previous_background":previous_background,
                            "chat_context": chat_context,
                            "human_name": human_name,
                            "gpt_name": gpt_name
                        })
                    }
    ]
    return messages

def history_to_messages(history: History, system: str) -> Messages:
    messages = [{'role': "system", 'content': system}]
    for h in history:
        messages.append({'role': "user", 'content': h[0]})
        messages.append({'role': "assistant", 'content':
            h[1]
        })
    return messages

def messages_to_history(messages: Messages) -> Tuple[str, History]:
    assert messages[0]['role'] == "system"
    system = messages[0]['content']
    history = []
    import numpy as np
    import pandas as pd
    from copy import deepcopy
    messages = deepcopy(messages)

    messages_ = []
    for ele in messages[1:]:
        if not messages_:
            messages_.append(ele)
        else:
            if messages_[-1]["role"] == ele["role"]:
                continue
            else:
                messages_.append(ele)

    last_message = messages_[-1]
    last_role = last_message["role"]
    if last_role == "user":
        messages_.append(
            {
                "role": "assistant",
                "content": ""
            }
        )
    history = pd.DataFrame(np.asarray(messages_).reshape([-1, 2]).tolist()).applymap(
        lambda x: x["content"]
    ).applymap(
        lambda x: x
    ).values.tolist()
    return system, history

def qwen_hf_predict_stream(messages, qw_model = qw_model,
    tokenizer = tokenizer, streamer = iter_streamer,
    do_sample = True,
    top_p = 0.95,
    top_k = 40,
    max_new_tokens = 2070,
    max_input_length = 3500,
    temperature = 0.9,
    repetition_penalty = 1.0,
    device = "cuda"):

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt",
        add_generation_prompt=True
    )
    model_inputs = encodeds.to(device)

    def generate_and_signal_complete():
        qw_model.generate(model_inputs, max_new_tokens=max_new_tokens,
                                    do_sample=do_sample,
                                      streamer = streamer,
                                      top_p = top_p,
                                      top_k = top_k,
                                      temperature = temperature,
                                      repetition_penalty = repetition_penalty,
                                      )

    t1 = Thread(target=generate_and_signal_complete)
    t1.start()

    system, history = messages_to_history(messages)

    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
        history[-1][1] = partial_text
        yield system, history
    #out = tokenizer.batch_decode(generated_ids)[0].split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()

#### return background iter
def init_background_chat(single_name_1, single_name_2):
    if hasattr(single_name_1, "value"):
        single_name_1_ = single_name_1.value
    else:
        single_name_1_ = single_name_1
    if hasattr(single_name_2, "value"):
        single_name_2_ = single_name_2.value
    else:
        single_name_2_ = single_name_2

    messages = init_background_instruction(single_name_1_, single_name_2_)
    req_iter = qwen_hf_predict_stream(messages)
    for system, history in req_iter:
        yield history[-1][1]

#### return new background iter
def new_background_chat(human_name:str, gpt_name:str, previous_background:str,
    history: Optional[History]):

    if history is None:
        history = []
    #### drop empty system message head
    hist_chat_messages = history_to_messages(history, "")[1:]

    messages = new_background_instruction(human_name, gpt_name, previous_background,
        hist_chat_messages)
    req_iter = qwen_hf_predict_stream(messages)
    for system, history in req_iter:
        yield history[-1][1]

#### return system hist iter
def model_chat(query: Optional[str], history: Optional[History],
    human_name: str, gpt_name: str, background: str
) -> Tuple[str, str, History]:
    system_head_messages = init_chat_system_messages(human_name, gpt_name, background)
    assert type(system_head_messages) == type([])
    assert len(system_head_messages) == 1

    if query is None:
        query = ''
    if history is None:
        history = []

    #### drop empty system message head
    messages = history_to_messages(history, "")[1:]
    messages = system_head_messages + messages
    messages.append({'role': "user", 'content': query})

    req_iter = qwen_hf_predict_stream(messages)
    for system, history in req_iter:
        yield "", history

with gr.Blocks() as demo:
    gr.Markdown("""<center><font size=8>😊 Genshin Impact Qwen-1.5-7B-Chat Sharegpt Roleplay Turned Transformer Bot 🔥</center>""")

    with gr.Row():
        human_name = gr.Dropdown(choices = info_df["title"].values.tolist(), label = "🎩你的角色",
            interactive = True, value = "丽莎"
        )
        gpt_name = gr.Dropdown(choices = info_df["title"].values.tolist(), label = "🤖机器人角色",
            interactive = True, value = "提纳里"
        )
    with gr.Row():
        background_value = list(init_background_chat(human_name, gpt_name))[-1]
        background = gr.Textbox(background_value ,
            label = "🖼️对话背景（可编辑）", interactive = True, lines = 2)

    chatbot = gr.Chatbot(label='svjack/Genshin_Impact_Qwen_1_5_Chat_sharegpt_roleplay_chat_lora_small')
    textbox = gr.Textbox(lines=2, label='Input')

    with gr.Row():
        clear_history = gr.Button("🧹 清空历史")
        sumbit = gr.Button("🚀 发送")
    with gr.Row():
        reset_background = gr.Button("♻️重置对话背景")
        new_background = gr.Button("➡️推进对话背景")

    sumbit.click(model_chat,
                 inputs=[textbox, chatbot, human_name, gpt_name, background],
                 outputs=[textbox, chatbot],
                 concurrency_limit = 100)
    clear_history.click(fn=lambda _ : ("", []),
                        inputs=[],
                        outputs=[textbox, chatbot])

    reset_background.click(
        init_background_chat,
        inputs = [human_name, gpt_name],
        outputs = background
    )
    new_background.click(
        new_background_chat,
        inputs = [human_name, gpt_name, background, chatbot],
        outputs = background
    )

    human_name.change(
        init_background_chat,
        inputs = [human_name, gpt_name],
        outputs = background
    )
    human_name.change(fn=lambda _ : ("", []),
                        inputs=[],
                        outputs=[textbox, chatbot])
    gpt_name.change(
        init_background_chat,
        inputs = [human_name, gpt_name],
        outputs = background
    )
    gpt_name.change(fn=lambda _ : ("", []),
                        inputs=[],
                        outputs=[textbox, chatbot])

demo.queue(api_open=False)
demo.launch(max_threads=30, share = True)
