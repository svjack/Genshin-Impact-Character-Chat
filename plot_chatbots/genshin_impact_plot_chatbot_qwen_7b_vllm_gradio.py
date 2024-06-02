'''
peft
transformers
bitsandbytes
ipykernel
rapidfuzz
datasets
gradio
sentencepiece
vllm
openai

pip install peft transformers bitsandbytes ipykernel rapidfuzz datasets gradio sentencepiece vllm openai

A4000 x 2

python -m vllm.entrypoints.openai.api_server --model svjack/Genshin_Impact_Qwen_1_5_Plot_Chat_roleplay_chat_AWQ --dtype auto \
 --api-key token-abc123 --tensor-parallel-size 2 --quantization awq --max-model-len 4000 --gpu-memory-utilization 0.35 --port 8000

python -m vllm.entrypoints.openai.api_server --model svjack/DPO_Genshin_Impact_Qwen_1_5_Plot_Engine_Step_Json_Short_AWQ --dtype auto \
 --api-key token-abc123 --tensor-parallel-size 2 --quantization awq --max-model-len 2000 --gpu-memory-utilization 0.35 --port 8001
'''

import gradio as gr

from transformers import TextStreamer, TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import pandas as pd
import json
import numpy as np
import re
from datasets import Dataset, load_dataset
from rapidfuzz import fuzz
from threading import Thread

from openai import OpenAI

from typing import List, Optional, Tuple, Dict
History = List[Tuple[str, str]]
Messages = List[Dict[str, str]]

adapter_name_client_dict = {
    "chat": {
    "model": "svjack/Genshin_Impact_Qwen_1_5_Plot_Chat_roleplay_chat_AWQ",
    "client": OpenAI(
        api_key="token-abc123",
        base_url="http://localhost:8000/v1",
    )},
    "engine": {
    "model": "svjack/DPO_Genshin_Impact_Qwen_1_5_Plot_Engine_Step_Json_Short_AWQ",
    "client": OpenAI(
        api_key="token-abc123",
        base_url="http://localhost:8001/v1",
    )}
}

def openai_predict(messages,
    adapter_name = "chat",
    temperature = 0.01
    ):
    stream = adapter_name_client_dict[adapter_name]["client"].chat.completions.create(
            model=adapter_name_client_dict[adapter_name]["model"],  # Model name to use
            messages=messages,  # Chat history
            temperature=temperature,  # Temperature for text generation
            stream=True,  # Stream response
    )
    # Read and return generated text from response stream
    partial_message = ""
    for chunk in stream:
        #clear_output(wait = True)
        partial_message += (chunk.choices[0].delta.content or "")
        #print(partial_message)
    return partial_message
    #return out

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

def openai_predict_stream(messages,
    temperature = 0.01,
    adapter_name = "chat"
    ):
    stream = adapter_name_client_dict[adapter_name]["client"].chat.completions.create(
            model=adapter_name_client_dict[adapter_name]["model"],  # Model name to use
            messages=messages,  # Chat history
            temperature=temperature,  # Temperature for text generation
            stream=True,  # Stream response
    )

    system, history = messages_to_history(messages)

    # Read and return generated text from response stream
    partial_message = ""
    for chunk in stream:
        #clear_output(wait = True)
        partial_message += (chunk.choices[0].delta.content or "")
        #print(partial_message)
        history[-1][1] = partial_message
        yield system, history

    '''
    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
        history[-1][1] = partial_text
        yield system, history
    '''

def run_step_infer_times(x, times = 5, temperature = 0.01,
                        repetition_penalty = 1.0,
                        sim_val = 70,
                        adapter_name = "engine"
                        ):
    req = []
    for _ in range(times):
        #clear_output(wait = True)
        '''
        out = qwen_hf_predict([
                {
                    "role": "system",
                    "content": ""
                },
                {
                    "role": "user",
                    "content": x
                },
            ],
            repetition_penalty = repetition_penalty,
            temperature = temperature,
            max_new_tokens = 2070,
            max_input_length = 6000,
            adapter_name = adapter_name
        )
        '''
        out = openai_predict([
                {
                    "role": "system",
                    "content": ""
                },
                {
                    "role": "user",
                    "content": x
                },
            ],
            temperature = temperature,
            adapter_name = adapter_name
        )
        if req:
            val = max(map(lambda x: fuzz.ratio(x, out), req))
            #print(val)
            #print(req)
            if val < sim_val:
                req.append(out.strip())
            x = x.strip() + "\n" + out.strip()
        else:
            req.append(out.strip())
            x = x.strip() + "\n" + out.strip()
    return req

plot_summary_df = load_dataset("svjack/Genshin-Impact-Plot-Summary")["train"].to_pandas()
character_portrait_df = load_dataset("svjack/Genshin-Impact-Plot-Character-Portrait-Merged")["train"].to_pandas()
#character_portrait_df = load_dataset("svjack/Genshin-Impact-Plot-Character-Portrait")["train"].to_pandas()
del character_portrait_df["portrait"]
plot_summary_df = plot_summary_df.dropna().drop_duplicates()
character_portrait_df = character_portrait_df.dropna().drop_duplicates()

name_list = character_portrait_df["name"].dropna().drop_duplicates().values.tolist()
title_list = plot_summary_df["title"].dropna().drop_duplicates().values.tolist()

ll0 = ['捕风的异乡人',
 '为了没有眼泪的明天',
 '巨龙与自由之歌',
 '浮世浮生千岩间',
 '辞行久远之躯',
 '迫近的客星',
 '我们终将重逢',
 '振袖秋风问红叶',
 '不动鸣神，恒常乐土',
 '无念无想，泡影断灭',
 '千手百眼，天下人间',
 '回响渊底的安魂曲',
 '穿越烟帷与暗林',
 '千朵玫瑰带来的黎明',
 '迷梦与空幻与欺骗',
 '赤土之王与三朝圣者',
 '虚空鼓动，劫火高扬',
 '卡利贝尔',
 '风起鹤归',
 '危途疑踪',
 '倾落伽蓝']

ll1 = ['海盗秘宝',
 '风、勇气和翅膀',
 '麻烦的工作',
 '暗夜英雄的不在场证明',
 '卢皮卡的意义',
 '真正的宝物',
 '骑士团长的一日假期',
 '若你困于无风之地',
 '在此世的星空之外',
 '旅行者观察报告',
 '浪花不再归海',
 '蒙德食遇之旅',
 '江湖不问出处',
 '盐花',
 '匪石',
 '云之海，人之海',
 '槐柯胡蝶，傩佑之梦',
 '奈何蝶飞去',
 '棋生断处',
 '「医心」',
 '鹤与白兔的诉说',
 '如梦如电的隽永',
 '影照浮世风流',
 '须臾百梦',
 '兵戈梦去，春草如茵',
 '赤金魂',
 '鸣神御祓祈愿祭',
 '梧桐一叶落',
 '陌野不识故人',
 '拾星之旅',
 '当他们谈起今夜',
 '没有答案的课题',
 '沉沙归寂',
 '致智慧者',
 '余温',
 '归乡',
 '乌合的虚像',
 '「狮之血」',
 '被遗忘的怪盗',
 '往日留痕']

assert all(map(lambda x: x in title_list, ll0))
assert all(map(lambda x: x in title_list, ll1))
title_list = ll0 + ll1

def find_plot_info(title):
    return dict(filter(lambda t2: t2[1] ,plot_summary_df[
        plot_summary_df["title"] == title
    ][[
     '故事背景b',
     '事件起因b',
     '事件经过b',
     '事件反转b',
     '事件结束b',
     '事件意义b',
     '后续剧情b'
    ]].rename(
        columns = {
     '故事背景b': "故事背景",
     '事件起因b': "事件起因",
     '事件经过b': "事件经过",
     '事件反转b': "事件反转",
     '事件结束b': "事件结束",
     '事件意义b': "事件意义",
     '后续剧情b': "后续剧情"
        }
    ).T.apply(
        lambda x: sorted(set(
            map(lambda z: z.strip() ,filter(lambda y: y.strip() ,x.tolist()))
        ), key = len, reverse = True), axis = 1
    ).to_dict().items()))

#### top_k 不应该过度限制角色数量，如果是关键角色
def pick_names_from_summary(summary, name_list, top_k = 6):
    repeat_l = ["自己", "荧", "空", "彼此", "天君", "夜叉", "影"]
    name_list_in_x = list(filter(lambda y: y in summary and y not in repeat_l, name_list))
    return sorted(name_list_in_x, key = lambda x: len(summary.split(x)), reverse = True)[:top_k]

def build_instruction_for_plot_engine(title ,Summary, name_list_in_x):
    instruction = "\n".join(map(lambda y: y.strip() ,filter(lambda x: x.strip() ,'''
        故事标题:{title}
        故事背景:{Summary}
        参与角色:{all_candidates}
        '''.format(
            **{
                "title": title,
                "Summary": Summary.replace("\n", "").replace(" ", ""),
                "all_candidates": "、".join(name_list_in_x),
            }
        ).split("\n"))))
    return instruction

def build_chat_system_head(total_background, now_background, Person1, Person2, character_portrait_df):
    system_head = '''
    故事背景:{total_background}
    当前故事背景:{now_background}
    参与者1:{Person1}
    参与者1角色经历:{Person1_hist}
    参与者1性格特征:{Person1_char}
    参与者1剧情中的作用:{Person1_plot}
    参与者2:{Person2}
    参与者2角色经历:{Person2_hist}
    参与者2性格特征:{Person2_char}
    参与者2剧情中的作用:{Person2_plot}
    要求进行"{Person1}"与"{Person2}"之间的对话。
    我扮演"{Person1}"，你扮演"{Person2}"。
    '''.format(
        **{
            "total_background": total_background.replace("\n", "").replace(" ", ""),
            "now_background": now_background.replace("\n", "").replace(" ", ""),
            "Person1":Person1,
            "Person1_hist": dict(character_portrait_df[["name", "角色经历"]].values.tolist()).get(Person1, "").replace("\n", "").replace(" ", ""),
            "Person1_char": dict(character_portrait_df[["name", "性格特征"]].values.tolist()).get(Person1, "").replace("\n", "").replace(" ", ""),
            "Person1_plot": dict(character_portrait_df[["name", "剧情中的作用"]].values.tolist()).get(Person1, "").replace("\n", "").replace(" ", ""),
            "Person2":Person2,
            "Person2_hist": dict(character_portrait_df[["name", "角色经历"]].values.tolist()).get(Person2, "").replace("\n", "").replace(" ", ""),
            "Person2_char": dict(character_portrait_df[["name", "性格特征"]].values.tolist()).get(Person2, "").replace("\n", "").replace(" ", ""),
            "Person2_plot": dict(character_portrait_df[["name", "剧情中的作用"]].values.tolist()).get(Person2, "").replace("\n", "").replace(" ", ""),
        }
    )
    system_head = "\n".join(map(lambda y: y.strip() ,filter(lambda x: x.strip() ,system_head.split("\n"))))
    return system_head

def run_step_infer_times_run(x, times = 5, temperature = 0.01,
                        repetition_penalty = 1.0,
                        sim_val = 70,
                        name_list_in_x = [],
                        adapter_name = "engine"
                        ):
    req = []
    for _ in range(times):
        #clear_output(wait = True)
        out = openai_predict([
                {
                    "role": "system",
                    "content": ""
                },
                {
                    "role": "user",
                    "content": x
                },
            ],
            temperature = temperature,
            adapter_name = adapter_name
        )
        try:
            d = eval(out)
            assert len(d) == 3
            assert "参与者1" in d and "参与者2" in d and "当前故事背景" in d
            d["当前故事背景"] = d["当前故事背景"].replace("\n", "").replace(" ", "")
            out = str(d)
        except:
            print("parse error")
            continue
        if req:
            val = max(map(lambda x: fuzz.ratio(x, out), req))
            if val < sim_val:
                if d["参与者1"] != d["参与者2"]:
                    req.append(out.strip())
                    req_iter = list(map(eval ,req))
                    yield "\n".join(map(lambda d:
                        "\n".join(map(lambda t2: "{}:{}".format(t2[0], t2[1]) ,d.items()))
                     ,req_iter))
                else:
                    if name_list_in_x:
                        l = list(
                        filter(lambda ele: ele not in [d["参与者1"], d["参与者2"]] and ele in d["当前故事背景"],
                        name_list_in_x)
                        )
                        if l:
                            sc = d["当前故事背景"]
                            rp = d["当前故事背景"].replace(d["参与者2"], l[0])
                            assert sc in out
                            out = out.replace(sc, rp)
                            req.append(out.strip())
                            req_iter = list(map(eval ,req))
                            req_iter = list(filter(lambda x: type(x) == type({}), req_iter))
                            yield "\n".join(map(lambda d:
                                "\n".join(map(lambda t2: "{}:{}".format(t2[0], t2[1]) ,d.items()))
                             ,req_iter))
        else:
            if d["参与者1"] != d["参与者2"]:
                req.append(out.strip())
                req_iter = list(map(eval ,req))
                req_iter = list(filter(lambda x: type(x) == type({}), req_iter))
                yield "\n".join(map(lambda d:
                    "\n".join(map(lambda t2: "{}:{}".format(t2[0], t2[1]) ,d.items()))
                 ,req_iter))
            else:
                if name_list_in_x:
                    l = list(
                        filter(lambda ele: ele not in [d["参与者1"], d["参与者2"]] and ele in d["当前故事背景"],
                        name_list_in_x)
                        )
                    if l:
                        sc = d["当前故事背景"]
                        rp = d["当前故事背景"].replace(d["参与者2"], l[0])
                        assert sc in out
                        out = out.replace(sc, rp)
                        req.append(out.strip())
                        req_iter = list(map(eval ,req))
                        req_iter = list(filter(lambda x: type(x) == type({}), req_iter))
                        yield "\n".join(map(lambda d:
                            "\n".join(map(lambda t2: "{}:{}".format(t2[0], t2[1]) ,d.items()))
                         ,req_iter))
        x = x.strip() + "\n" + out.strip()
    req_iter = list(map(eval ,req))
    req_iter = list(filter(lambda x: type(x) == type({}), req_iter))
    yield "\n".join(map(lambda d:
        "\n".join(map(lambda t2: "{}:{}".format(t2[0], t2[1]) ,d.items()))
     ,req_iter))

def get_global_background(title, content_option):
    if hasattr(title, "value"):
        title_ = title.value
    else:
        title_ = title
    if hasattr(content_option, "value"):
        content_option_ = content_option.value
    else:
        content_option_ = content_option
    plot_info_dict = find_plot_info(title_)
    #list(plot_info_dict.keys())
    #['故事背景', '事件起因', '事件经过', '事件反转', '事件结束', '事件意义', '后续剧情']
    Summary = plot_info_dict.get(content_option_, [""])[0]
    Summary = Summary.replace("\n\n", "\n")
    return Summary

def trigger_current_background(title ,Summary, current_background_num):
    if hasattr(title, "value"):
        title_ = title.value
    else:
        title_ = title
    if hasattr(Summary, "value"):
        Summary_ = Summary.value
    else:
        Summary_ = Summary
    if hasattr(current_background_num, "value"):
        current_background_num_ = current_background_num.value
    else:
        current_background_num_ = current_background_num
    current_background_num_ = int(current_background_num_)
    name_list_in_x = pick_names_from_summary(Summary_, name_list)
    #name_list_in_x
    instruction = build_instruction_for_plot_engine(title_ ,Summary_, name_list_in_x)
    #print(instruction)
    out_l_iter = run_step_infer_times_run(instruction, times = current_background_num_,
        name_list_in_x = name_list_in_x, temperature = 0.01)
    for ele in out_l_iter:
        yield ele

def trigger_selected_current_background(current_background, select_index, global_background):
    if hasattr(current_background, "value"):
        current_background_ = current_background.value
    else:
        current_background_ = current_background
    if hasattr(select_index, "value"):
        select_index_ = select_index.value
    else:
        select_index_ = select_index
    if hasattr(global_background, "value"):
        global_background_ = global_background.value
    else:
        global_background_ = global_background
    #select_index_ = int(select_index_)
    assert type(select_index_) == type("")

    l = current_background_.split("\n")
    req = []
    for i, ele in enumerate(l):
        i_res_3 = i % 3
        if not req:
            req.append([ele])
        else:
            if i_res_3 == 0:
                req.append([ele])
            elif i_res_3 == 1:
                req[-1].append(ele)
            else:
                req[-1].append(ele)
    req = list(filter(lambda x: len(x) == 3, req))
    if not req:
        return ["", "", ""]
    select_index_num = min(len(req) - 1, int(select_index_.split(":")[-1]))
    if select_index_.startswith("当前对话背景"):
        return list(map(lambda x: ":".join(x.split(":")[1:]) if ":" in x else x,
            req[select_index_num]
        ))
    else:
        t3 = list(map(lambda x: ":".join(x.split(":")[1:]) if ":" in x else x,
            req[select_index_num]
        ))
        t3[-1] = global_background_
        return t3

def model_chat(query: Optional[str], history: Optional[History],
    Person1:str, Person2: str, background:str, selected_current_background:str,
    role_want_to_play: str
) -> Tuple[str, str, History]:
    #system_head_messages = init_chat_system_messages(human_name, gpt_name, background)
    assert role_want_to_play in ['对话者1', '对话者2']
    d = {}
    if role_want_to_play == "对话者1":
        d["参与者1"] = Person1
        d["参与者2"] = Person2
    else:
        d["参与者1"] = Person2
        d["参与者2"] = Person1
    system_head = build_chat_system_head(
        background, selected_current_background,
        d["参与者1"], d["参与者2"],
        character_portrait_df)
    system_head_messages = [
        {
            "role": "system",
            "content": system_head
        }
    ]
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

    print("messages :")
    print(messages)

    import re
    zh_pattern = u"[\u4e00-\u9fa5]+"

    req_iter = openai_predict_stream(messages, temperature = 0.5)
    for system, history in req_iter:
        if history:
            history[-1][-1] = history[-1][-1].replace(d["参与者2"], "")
            zh_l = re.findall(zh_pattern, history[-1][-1])
            if zh_l:
                history[-1][-1] = history[-1][-1][history[-1][-1].find(zh_l[0]):]
        yield "", history

def re_model_chat(query: Optional[str], history: Optional[History],
    Person1:str, Person2: str, background:str, selected_current_background:str,
    role_want_to_play: str
) -> Tuple[str, str, History]:
    if len(history) >= 1:
        query, resp = history[-1]
        req_iter = model_chat(query, history[:-1], Person1, Person2, background, selected_current_background, role_want_to_play)
        for system, history in req_iter:
            yield "", history
    else:
        yield "", []


with gr.Blocks() as demo:
    gr.Markdown("""<center><font size=8>😁 Genshin Impact Qwen-1.5-7B-Chat Plot Roleplay Turned ⚡️ vLLM ⚡️ Bot 🔥</center>""")

    with gr.Row():
        with gr.Column():
            with gr.Row():
                title = gr.Dropdown(choices = title_list,
                    label = "💡章节标题",
                    interactive = True, value = title_list[0]
                )
            with gr.Row():
                content_option = gr.Radio(['故事背景', '事件起因', '事件经过', '事件反转', '事件结束', '事件意义', '后续剧情'],
                    label="✨章节内容", interactive = True, value = "故事背景")
            with gr.Column():
                global_background_value = get_global_background(title, content_option)
                global_background = gr.Textbox(global_background_value,
                    label = "🌍全局对话背景（可编辑）", interactive = True)
            with gr.Column():
                current_background_num = gr.Slider(
                    1, 10, value=5, label="生成当前对话背景个数", step = 1
                )
                current_background_button = gr.Button("⚙️重新生成当前对话背景")
                current_background_value = list(trigger_current_background(title ,global_background, current_background_num))[-1]
                #print("current_background_value :", current_background_value)

                current_background = gr.Textbox(current_background_value,
                    label = "🖼️生成的可选当前对话背景（可编辑），从右侧进行选择选用。", interactive = True)
        with gr.Column():
            with gr.Row():
                chat_index_list = []
                for i in range(10):
                    for prefix in ["全局对话背景", "当前对话背景"]:
                        chat_index_list.append("{}:{}".format(prefix, i))

                chat_index = gr.Dropdown(choices = chat_index_list,
                    label = "✏️选定对话索引",
                    interactive = True, value = chat_index_list[0]
                )
                Person1_val, Person2_val, selected_current_background_value = trigger_selected_current_background(current_background, chat_index, global_background)
                Person1 = gr.Textbox(Person1_val,
                    label = "🤭对话者1", interactive = False)
                Person2 = gr.Textbox(Person2_val,
                    label = "😊对话者2", interactive = False)
                person_option = gr.Radio(['对话者1', '对话者2'],
                    label="☑️你想扮演的对话者", interactive = True, value = "对话者1")
            with gr.Row():
                selected_current_background = gr.Textbox(selected_current_background_value,
                    label = "📚选定对话背景（可编辑）", interactive = True)

            chatbot = gr.Chatbot(
                label='svjack/Genshin_Impact_Qwen_1_5_Plot_Chat_roleplay_chat_AWQ/DPO_Genshin_Impact_Qwen_1_5_Plot_Engine_Step_Json_Short_AWQ',
                height = 768
                )
            textbox = gr.Textbox(lines=2, label='Input')

            with gr.Row():
                sumbit = gr.Button("🚀 发送")
            with gr.Row():
                re_sumbit = gr.Button("♻️🚀 重新发送")
                clear_history = gr.Button("🧹 清空历史")

    title.change(
        get_global_background,
        [title, content_option],
        global_background
    )
    content_option.change(
        get_global_background,
        [title, content_option],
        global_background
    )
    global_background.change(
        trigger_current_background,
        [title ,global_background, current_background_num],
        current_background
    )
    current_background_num.change(
        trigger_current_background,
        [title ,global_background, current_background_num],
        current_background
    )
    current_background_button.click(
        trigger_current_background,
        [title ,global_background, current_background_num],
        current_background
    )
    chat_index.change(
        trigger_selected_current_background,
        [current_background, chat_index, global_background],
        [Person1, Person2 ,selected_current_background]
    )
    current_background.change(
        trigger_selected_current_background,
        [current_background, chat_index, global_background],
        [Person1, Person2 ,selected_current_background]
    )

    sumbit.click(model_chat,
                 inputs=[textbox, chatbot, Person1, Person2,
                 global_background, selected_current_background, person_option],
                 outputs=[textbox, chatbot],
                 concurrency_limit = 100)
    re_sumbit.click(
                 re_model_chat,
                 inputs=[textbox, chatbot, Person1, Person2,
                 global_background, selected_current_background, person_option],
                 outputs=[textbox, chatbot],
                 concurrency_limit = 100)

    chat_index.change(
        fn=lambda _ : ("", []),
                            inputs=[],
                            outputs=[textbox, chatbot])
    current_background.change(
        fn=lambda _ : ("", []),
                            inputs=[],
                            outputs=[textbox, chatbot])
    selected_current_background.change(
        fn=lambda _ : ("", []),
                            inputs=[],
                            outputs=[textbox, chatbot])
    Person1.change(
        fn=lambda _ : ("", []),
                            inputs=[],
                            outputs=[textbox, chatbot])
    Person2.change(
        fn=lambda _ : ("", []),
                            inputs=[],
                            outputs=[textbox, chatbot])
    person_option.change(
        fn=lambda _ : ("", []),
                            inputs=[],
                            outputs=[textbox, chatbot])
    clear_history.click(fn=lambda _ : ("", []),
                        inputs=[],
                        outputs=[textbox, chatbot])

demo.queue(api_open=False)
demo.launch(max_threads=30, share = True)
