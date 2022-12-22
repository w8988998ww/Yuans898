import gradio as gr
import os
os.system('cd monotonic_align && python setup.py build_ext --inplace && cd ..')

import time
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import re
import langid
import jieba
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence, cleaned_text_to_sequence
from text.cleaners import japanese_cleaners
from scipy.io.wavfile import write

def getMixText(text):
    langid.set_languages(['zh','en'])
    seg_list = jieba.cut(text, cut_all=False)
    clean_list=[]
    for seg in seg_list:
        langtext='[ZH]'
        if(len(seg)>0):
            lang=langid.classify(seg)[0]
            if lang == 'en':
                langtext='[EN]'
            elif lang=='zh':
                langtext='[ZH]'
            elif lang=='ja':
                langtext='[JA]'
            clean_list.append(langtext+seg+langtext)
    return ''.join(clean_list)
    
def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps_ms = utils.get_hparams_from_file("save_model/config2.json")
hps = utils.get_hparams_from_file("save_model/config2.json")
net_g_ms = SynthesizerTrn(
    len(symbols),
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps.data.hop_length,
    n_speakers=hps_ms.data.n_speakers,
    **hps_ms.model)

npclists=[]
with open("save_model/npclists.txt",'r') as r:
    for npc in r.readlines():
        npclists.append(npc.split('|')[-1])
        print(npc)
r.close

def tts(spkid, text):
    if(len(re.findall(r'\[ZH\].*?\[ZH\]', text))==0 and len(re.findall(r'\[EN\].*?\[EN\]', text))==0 and len(re.findall(r'\[JA\].*?\[JA\]', text))==0):
                text=getMixText(text)
    sid = torch.LongTensor([spkid])  # speaker identity
    stn_tst = get_text(text, hps_ms)

    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        t1 = time.time()
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][
            0, 0].data.float().numpy()
        t2 = time.time()
    return "成功,耗时"+str((t2-t1))+"s", (hps.data.sampling_rate, audio)

_ = utils.load_checkpoint("save_model/G_181400.pth", net_g_ms, None)

def clean_text(text):
    return japanese_cleaners(text)

app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("Basic"):
            tts_input1 = gr.TextArea(label="在这输入文字", value="在我的后园，可以看见墙外有两株树，一株是枣树，还有一株也是枣树。 这上面的夜的天空，奇怪而高，我生平没有见过这样的奇怪而高的天空。 他仿佛要离开人间而去，使人们仰面不再看见。 然而现在却非常之蓝，闪闪地眨着几十个星星的眼，冷眼。 他的口角上现出微笑，似乎自以为大有深意，而将繁霜洒在我的园里的野花草上。 我不知道那些花草真叫什么名字，人们叫他们什么名字。 我记得有一种开过极细小的粉红花，现在还开着，但是更极细小了，她在冷的夜气中，瑟缩地做梦，梦见春的到来，梦见秋的到来，梦见瘦的诗人将眼泪擦在她最末的花瓣上，告诉她秋虽然来，冬虽然来，而此后接着还是春，胡蝶乱飞，蜜蜂都唱起春词来了。 她于是一笑，虽然颜色冻得红惨惨地，仍然瑟缩着。 枣树，他们简直落尽了叶子。")
            tts_input2 = gr.Dropdown(label="人物", choices=npclists, type="index", value=npclists[0])
            tts_submit = gr.Button("合成", variant="primary")
            tts_output1 = gr.Textbox(label="信息")
            tts_output2 = gr.Audio(label="结果")
            tts_submit.click(tts, [tts_input2, tts_input1], [tts_output1, tts_output2])
    app.launch()