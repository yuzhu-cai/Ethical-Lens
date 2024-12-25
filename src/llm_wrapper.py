import openai
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import PeftModel
import re

class LocalLLM(object):
    def __init__(self, model_name, device):
        self.model_name = model_name
        if "vicuna" in model_name.lower():
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_4bit=True, device_map={"":device})
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.temperature = 0
        self.max_new_tokens = 1024
    
    def generate(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(
            **input_ids, 
            do_sample=True,
            temperature=self.temperature, 
            max_new_tokens=self.max_new_tokens
        )
        outputs = self.tokenizer.decode(generate_ids[0][len(input_ids["input_ids"][0]):], skip_special_tokens=True, spaces_between_special_tokens=False)
        return outputs


class LocalChatLLM(object):
    def __init__(self, model_name):
        self.model_name = model_name
        if "yi" in model_name.lower():
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_4bit=True, device_map="auto")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model_max_length = self.tokenizer.model_max_length
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = 2000

    def generate(self, messages):
        # input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')
        output_ids = self.model.generate(input_ids.to(self.device), do_sample=False, temperature=0.0)
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response



class LocalChatLLM_llama(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_4bit=True, device_map="auto")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model_max_length = self.tokenizer.model_max_length
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = 2000

    def generate(self, messages):
        input_ids = self.tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        output_ids = self.model.generate(input_ids.to(self.device), do_sample=False, temperature=0.0)
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response

class LocalChatLLM_qwen(object):
    def __init__(self, model_name):
        from modelscope import AutoModelForCausalLM, AutoTokenizer
        from modelscope import GenerationConfig
        self.model_name = model_name # Qwen/Qwen-7B-Chat
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", trust_remote_code=True, fp16=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(self.model_name, trust_remote_code=True)

        self.model_max_length = self.tokenizer.model_max_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = 2000

    def generate(self, messages):
        history = []
        for i in range(len(messages)-1):
            if messages[i]["role"] == "system" or messages[i]["role"] == "user":
                if messages[i+1]["role"] == "assistant":
                    history.append((messages[i]["content"],messages[i+1]["content"]))
                else:
                    history.append((messages[i]["content"],"ok"))
        # import ipdb; ipdb.set_trace()
        response, history = self.model.chat(self.tokenizer, messages[-1]["content"], history=history)
        return response

class LocalChatLLM_glm(object):
    def __init__(self, model_name):
        self.model_name = model_name # "THUDM/chatglm-6b"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, device_map="auto", trust_remote_code=True).half().eval()
        
        self.model_max_length = self.tokenizer.model_max_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = 2000

    def generate(self, messages):
        history = []
        for i in range(len(messages)-1):
            if messages[i]["role"] == "system" or messages[i]["role"] == "user":
                if messages[i+1]["role"] == "assistant":
                    history.append((messages[i]["content"],messages[i+1]["content"]))
                else:
                    history.append((messages[i]["content"],"ok"))
        
        response = self.model.chat(self.tokenizer, messages[-1]["content"], history=history)
        return response

class OpenAILLM(object):
    def __init__(self, api_key, model_name="gpt-4-turbo-preview"):
        self.api_base = "https://api.openai-proxy.com/v1/"
        self.api_key = api_key
        self.model_name = model_name

    def generate(self, messages):
        openai.api_base = self.api_base
        openai.api_key = self.api_key
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=256,
                    top_p=0.7,
                    frequency_penalty=0,
                    presence_penalty=0
                )["choices"][0]['message']['content']
                return response
            except:
                print("Network error, retrying in 5 seconds...")
                time.sleep(5)
    
    def get_revision(self, text):
        try:
            result = re.search(r'@@@ Revision: (.*?)\s*@@@', text).group(1).replace("\n", "")
        except:
            result = "<no revision>"
        return result

    def get_score(self, text):
        try:
            result = re.search(r'@@@ Score: (.*?)\s*@@@', text).group(1).replace("\n", "")
        except:
            result = "<no score>"
        return result

if __name__ == "__main__":
    llm = LocalChatLLM_qwen("/home/ubuntu/yuxiwei/llm_finetune/Qwen-7B-Chat", "/home/ubuntu/yuxiwei/Qwen/output/merged_all_0223")
    # llm1 = LocalChatLLM_qwen("/home/ubuntu/yuxiwei/llm_finetune/Qwen-7B-Chat", "/home/ubuntu/yuxiwei/Qwen/output/merged_r2/checkpoint-900")
    # llm = LocalChatLLM("/dev/shm/data/shengyin/pretrain_model/Yi/Yi")
    # from temp import openai_messages_spilit_alternate
    api_key = ""
    model = "gpt-4-turbo-preview"
    # llm = OpenAILLM(api_key, model_name=model)

    import json
    data = []
    with open('/home/ubuntu/DATA1/yuzhucai/data_prepare/tab/100/records_100.json', 'r') as file:
        data = json.load(file)

    from temp import openai_messages_spilit_alternate, expert_revision
    # k = 0
    for i in range(50):
        expert_revision[-1]['content'] = f"@@@ Prompt: {data[i]['prompt']} \n@@@ Issue: {data[i]['type']}\n@@@ Reason: Due to the potential for {data[i]['type']} issues with images generated from this prompt, please modify it to ensure the resulting images are non-toxic and suitable for viewing. The problem may stem from descriptions of the image's style or elements(zombies, Vampires, etc) within the image that reference toxic content. Please make the necessary adjustments."
        response = llm.generate(expert_revision)
        print(response)
        print('============================')

    # for i in range(100):
    #     openai_messages_spilit_alternate[-1]['content'] = data[i]['prompt']
    #     response = llm.generate(openai_messages_spilit_alternate)
    #     print(response)
    #     print('============================')
    # >>> CUDA_VISIBLE_DEVICES=4,5,6 python llm_wrapper.py