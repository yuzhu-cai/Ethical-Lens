import warnings
warnings.filterwarnings("ignore")

import re
import os
import torch
import random
import numpy as np

from PIL import Image
from diffusers import DiffusionPipeline

import sys
sys.path.append('./')
from src.llm_wrapper import OpenAILLM, LocalChatLLM, LocalChatLLM_qwen
from src.classifier import safe_classifier
from src.constants_bias import openai_judge_bias, openai_messages_spilt_bias, openai_messages_combine_bias, openai_judge_age
from src.constants_bias import gender_list, race_list, age_list
from src.temp import expert_revision, openai_messages_spilit_alternate
from src.fuzz import ClipFuzz
from src.random_select import RandomSelect
from common.AdaTrans.faceedit import Trans
from common.random_seed import prepare_seed
from evaluator.fairface_metric import FairFaceMetric

class Ethicallens():
    def __init__(self, cfg):
        rgi = cfg['relative_gpu_index']
        self.device = torch.device(f'cuda:{rgi}' if torch.cuda.is_available() else 'cpu')
        self.clipseg_path = cfg['clipseg_path']
        self.checkpoints = cfg['checkpoints']
        self.if_ff = cfg['if_fairface']
        self.resolution = cfg['resolution']

        # load models
        self.text2img = DiffusionPipeline.from_pretrained(cfg['text2img_path'], torch_dtype=torch.float16, from_tf=False).to(self.device)

        self.llm = LocalChatLLM_qwen(cfg['llm_path'], cfg['peft_path'])

        self.sc = safe_classifier(self.checkpoints)
        self.clipfuzz = ClipFuzz(model_path=self.clipseg_path, dilation_radius=30, blur_radius=0)
        self.fuziness = 0.2
        self.faceedit = Trans(cfg['faceedit_device'])
        self.random_selector = RandomSelect()
        # self.faceedit = None
        if self.if_ff:
            self.fairface = FairFaceMetric(cfg['fairface'])
        
        # for middle age
        self.other_middle = 'established adulthood'
        self.middle_age_list = []

    def get_text(self, text):
        try:
            explanation = re.search(r'@@@ Explanation: (.*?)\s*@@@', text).group(1).replace("\n", "")
        except:
            explanation = "<no explanation>"
        try:
            label = re.search(r'@@@ Label: (.*?)\s*@@@', text).group(1).replace("\n", "")
        except:
            label = "<no label>"
        try:
            text = re.search(r'@@@ Text: (.*?)$', text, re.DOTALL).group(1).strip()
        except:
            text = "<no text>"
        return text, label, explanation

    def get_people_and_explanation(self, response):
        '''
        people: a list of people who are involved in the prompt and have the same attributes, example: [], ['Taylor Swift', 'doctors', 'nurse']

        explanation: a dict, key is from people and value is a sub dict. 
        The sub dict has two keys: "type" and "bias". 
        "type" has 3 values which is 0, 1, 2. 0 is no bias, 1 is only one person and biased, 2 is more than one person and biased.
        The value of "bias" must involve 'gender', 'race', 'age'.
        example: {'Taylor Swift': {'type': 0, 'bias': []}, 'nurse': {'type':1, 'bias': ['gender', 'race', 'age']}, 'doctors': {'type':2, 'bias': ['race', 'age']} }

        '''
        people_pattern = r"@@@ People: (\[.*?\])"
        explanation_pattern = r"@@@ Explanation: (\{.*\})"

        # Using re.search to find matches and extract data
        try:
            people_match = re.search(people_pattern, response)
            people = eval(people_match.group(1)) if people_match else []
        except:
            people = []
        try:
            explanation_match = re.search(explanation_pattern, response)
            explanation = eval(explanation_match.group(1)) if explanation_match else {}
        except:
            explanation = {}

        return people, explanation

    def get_age(self, response):
        try:
            revision_match = re.search(r"@@@ Age: (\[.*?\])", response)
            revision = eval(revision_match.group(1)) if revision_match else ['infancy', 'childhood', 'adolescence', 'young adulthood', 'middle age', 'old age']
        except:
            revision = ['infancy', 'childhood', 'adolescence', 'young adulthood', 'middle age', 'old age']
        return revision
    
    def get_revision(self, response):
        try:
            revision_match = re.search(r"@@@ Revision: (\[.*?\])", response)
            revision = eval(revision_match.group(1)) if revision_match else None
        except:
            revision = "<no revision>"
        return revision

    def get_prompt(self, text):
        try:
            revision = re.search(r'@@@ Text: (.*?)$', text, re.DOTALL).group(1).strip()
        except:
            try:
                revision = re.search(r'@@@\n\nText: (.*?)$', text, re.DOTALL).group(1).strip()
            except:
                revision = "<no text>"
        try:
            explanation = re.search(r'@@@ Explanation: (.*?)\s*@@@', text, re.DOTALL).group(1).strip()
        except:
            explanation = "<no explanation>"
        return revision, explanation
    
    def tox_pre(self, prompt, img_number, data):
        openai_messages_spilit_alternate[-1]['content'] = prompt
        response = self.llm.generate(openai_messages_spilit_alternate)
        text, label, explanation = self.get_text(response)
        data["toxicity_pre_response"] = {
            "response": response,
            "text": text,
            "label":label,
            "explanation": explanation
        }
        if label == "<no label>" or text == '<no text>':
            text = prompt
            label = 'K0'
        if label == 'K3':
            image = Image.new('RGB', (512, 512), color=(0, 0, 0))
            data['images'] = []
            for i in range(img_number):
                data['images'].append(image)
            data['status'] = 'FAIL'
            return None, None
        
        # import pdb; pdb.set_trace()
        # print(label)
        return text, label
    
    def process_middle_age(self, selected_age):

        for idx, item in enumerate(selected_age):
            if item == 'middle age':
                self.middle_age_list.append(idx)
                continue

        return


    def bias_pre(self, text, img_number, data):
        # bias pre-process
        openai_messages_spilt_bias[-1]["content"] = text
        try:
            print("========bias gene++++++++")
            response = self.llm.generate(openai_messages_spilt_bias)
            people, explanation = self.get_people_and_explanation(response)
            data["bias_pre_response"] = {
                "response": response,
                "people":people,
                "explanation": explanation
            }
        except:
            people = []
            explanation = {}

        diverse_people = {}
        flag = False
        print(people)
        print(explanation)
        try:
            for cluster in people:
                type = explanation[cluster]['type']
                # print(f"type: {type}")
                if type != 0:
                    flag = True
        except:
            people = []
        # if people != []:
        if flag:
            # bias preprocess 2
            for cluster in people:
                type = explanation[cluster]['type']
                bias = explanation[cluster]['bias']

                if type == 0:
                    bias = []

                if 'age' in bias:
                    input_content = f"@@@ Keyword: {cluster}"
                    openai_judge_age[-1]["content"] = input_content

                    try:
                        age_response = self.llm.generate(openai_judge_age)
                        age_range = self.get_age(age_response)
                        data["bias_pre_response"].update({
                            "age_response": age_response,
                            "age_range": age_range
                        })
                    except:
                        age_range = ['infancy', 'childhood', 'adolescence', 'young adulthood', 'middle age', 'old age']
                else:
                    age_range = ['infancy', 'childhood', 'adolescence', 'young adulthood', 'middle age', 'old age']

                if type == 0:
                    diverse_people[cluster] = [cluster for i in range(img_number)]
                elif type == 1:
                    # gender + race + cluster + age
                    revision = [cluster for i in range(img_number)]
                    
                    selected_gender = self.random_selector.gender_forward(img_number)
                    selected_race = self.random_selector.race_forward(img_number)
                    selected_age = self.random_selector.age_forward(age_range, img_number)
                    print(selected_gender)
                    print(selected_race)
                    print(selected_age)
                    self.middle_age_list = []
                    self.process_middle_age(selected_age)

                    if 'gender' in bias:
                        revision = [selected_gender[i] + ' ' + revision[i] for i in range(img_number)]
                    if 'race' in bias:
                        revision = [selected_race[i] + ' ' + revision[i] for i in range(img_number)]
                    if 'age' in bias:
                        revision = [revision[i] + ' in ' + selected_age[i] for i in range(img_number)]
                    diverse_people[cluster] = revision
                elif type == 2:
                    
                    if 'age' in bias:
                        if len(age_range) == 1:
                            age_part = "with ages in " + age_range[0]
                        elif len(age_range) == 2:
                            age_part = f"with a range of ages from {age_range[0]} to {age_range[1]}"
                        else:
                            # import pdb; pdb.set_trace()
                            mid_part = ", ".join(age_range[1:-1])
                            age_part = f"with a range of ages from {age_range[0]} through {mid_part} to {age_range[-1]}"
                    else:
                        age_part = ""
                    
                    if 'race' in bias:
                        race_part = "a mix of White, Black, Latino-Hispanic, Asian and MiddleEastern"
                    else:
                        race_part = ""
                    
                    if 'gender' in bias:
                        gender_part = '50% male and 50% \female'
                    else:
                        gender_part = ""
                    
                    if bias == []:
                        suffix = ""
                    else:
                        suffix = ", consisting of "
                        if gender_part != "":
                            suffix = suffix + gender_part + ", and "
                        if race_part != "":
                            suffix = suffix + race_part + ", and "
                        if age_part != "":
                            if not suffix.endswith(", and "):
                                suffix = ", "
                            suffix = suffix + age_part
                        if suffix.endswith(", and "):
                            suffix = suffix[:-6]

                    diverse_people[cluster] = [cluster + suffix for i in range(img_number)]

            # bias preprocess 3
            revision, flag = [], True
            for k in range(img_number):
                tmp_diverse_people = {}
                for cluster in people:
                    tmp_diverse_people[cluster] = diverse_people[cluster][k]
                content_combine = f"@@@ Prompt_number: {1}\n@@@ Prompt: {text}\n@@@ diverse_people: {tmp_diverse_people}"
                openai_messages_combine_bias[-1]["content"] = content_combine

                try:
                    response_combine = self.llm.generate(openai_messages_combine_bias)
                    # print(response_combine)
                    tmp_revision = self.get_revision(response_combine)
                    # data["pre_response"]["response_combine"] = response_combine

                    if tmp_revision == "<no revision>":
                        tmp_revision = [text]
                    elif len(tmp_revision) > 1:
                        tmp_revision = tmp_revision[:1]
                except:
                    flag = False
                    tmp_revision = [text]
                revision.extend(tmp_revision)
            if flag:
                data["bias_pre_response"]["revision"] = revision
        else:
            # revision = [text for i in range(img_number)]
            revision = [text]
        
        return revision, people, explanation

    
    def tox_post_wo_bias(self, text, img_number, label, data):
        img_dict = {}
        raw_images = self.text2img(
            prompt=text,
            num_images_per_prompt=img_number,
            num_inference_steps=25,
            width=self.resolution,
            height=self.resolution,
            requires_safety_checker=False
        )['images']
        torch.cuda.empty_cache()
        issues = []
        for raw_image in raw_images:
            issues.append(self.sc.check(raw_image))
        
        bias_exp = data['bias_pre_response']['explanation']
        bias_people = data['bias_pre_response']['people']
        publics = []
        for person in bias_people:
            try:
                if bias_exp[person]['type'] == 0:
                    publics.append(person)
            except:
                print(f"bias_explanation: {bias_exp}")
                print(f"bias_people: {bias_people}")

        for i, issue in enumerate(issues):
            img_dict[f'img_{i}'] = {}
            num_inference_steps = 1
            img_dict[f'img_{i}'][f"round_{str(num_inference_steps)}"] = {}
            img_dict[f'img_{i}'][f"round_{str(num_inference_steps)}"]['issues'] = issue
            img_dict[f'img_{i}'][f"round_{str(num_inference_steps)}"]['prompt'] = text
            if "nsfw" in issue or "politic" in issue or "culture" in issue:
                expert_revision[-1]["content"] = f"@@@ Prompt: {text} \n@@@ Issue: {issue}\n@@@ Reason: Due to the potential for {issue} issues with images generated from this prompt, please modify it to ensure the resulting images are non-toxic and suitable for viewing. The problem may stem from descriptions of the image's style or elements(zombies, Vampires, etc) within the image that reference toxic content. Please make the necessary adjustments."
                opinion = self.llm.generate(expert_revision)
                text, explanation = self.get_prompt(opinion)
                img_dict[f'img_{i}'][f"round_{str(num_inference_steps)}"]['opinion'] = opinion
                img_dict[f'img_{i}'][f"round_{str(num_inference_steps)}"]['revision'] = text
                img_dict[f'img_{i}'][f"round_{str(num_inference_steps)}"]['explanation'] = explanation

                num_inference_steps += 1
                img_dict[f'img_{i}'][f"round_{str(num_inference_steps)}"] = {}
                raw_images[i] = self.text2img(
                            prompt=text,
                            num_inference_steps=25,
                            width=self.resolution,
                            height=self.resolution,
                            requires_safety_checker=False
                        )['images'][0]
                torch.cuda.empty_cache()
                issue = self.sc.check(raw_images[i])
                img_dict[f'img_{i}'][f"round_{str(num_inference_steps)}"]['issues'] = issue
                img_dict[f'img_{i}'][f"round_{str(num_inference_steps)}"]['prompt'] = text

                if "nsfw" in issue or "politic" in issue or "culture" in issue:
                    raw_images[i] = Image.new('RGB', (512, 512), color=(0, 0, 0))
                    data['status'] == 'FAIL'

            if "nude" in issue:
                raw_images[i] = self.clipfuzz.fuzzy(raw_images[i], "nude body", self.fuziness)
            if "public" in issue or 'A1' in label or publics != []:
                img_dict[f'img_{i}'][f"round_{str(num_inference_steps)}"]['public_figure'] = publics
                for peo in publics:
                    raw_images[i] = self.clipfuzz.fuzzy(raw_images[i], f"{peo}'s face", self.fuziness)

        print(img_dict)
        return raw_images, img_dict, publics


    def tox_post_w_bias(self, texts, label, data):
        raw_images = []
        img_dict = {}
        publics = []
        for id, text in enumerate(texts):
            raw_image, img_dict_ , pub = self.tox_post_wo_bias(text, 1, label, data)
            raw_images.append(raw_image[0])
            img_dict[f'img_{id}'] = img_dict_.pop("img_0")
            for item in pub:
                if item not in publics:
                    publics.append(item)
        return raw_images, img_dict, publics


    def face_edit(self, texts, raw_images, people, explanation, label, init_flag = True, second_id = 0):
        print("++++++++++++++++face edit++++++++++++")
        # raw_images = gen.face_edit(revision, raw_images, people, explanation)
        fairface = self.fairface
        faceedit = self.faceedit
        type_list = [explanation[group]['type'] for group in people]

        for id, raw_image in enumerate(raw_images):
            tmp_people, tmp_explanation = people, explanation
            if len(type_list) == 1 and type_list[0] == 1 and "middle age" in texts[id].lower():
            # if len(type_list) == 1 and type_list[0] == 1:
                if init_flag:
                    img_save_path = f'./results/tmp/_{id}.jpg'
                else:
                    img_save_path = f'./results/tmp/_second_middle_age_{second_id}.jpg'
                raw_image.save(img_save_path)
                tmp_data = {'image': {'path': img_save_path}}
                fairface_result = fairface.calculate(tmp_data)

                if fairface_result['pred']['people'] != 1:
                    continue
                else:
                    for key in fairface_result['pred']['gender']:
                        value = fairface_result['pred']['gender'][key]
                        if value == 1:
                            pred_gender = key
                            break

                    for key in fairface_result['pred']['age']:
                        value = fairface_result['pred']['age'][key]
                        if value == 1:
                            pred_age = key
                            break    
                        
                    if pred_age == '0-2':
                        pred_age = 'Infancy'
                    elif pred_age == '3-9':
                        pred_age = 'Childhood'
                    elif pred_age == '10-19':
                        pred_age = 'Adolescence'
                    elif pred_age == '20-29' or pred_age == '30-39':
                        pred_age = 'Young Adulthood'
                    elif pred_age == '40-49':
                        pred_age = 'Middle Age'
                    else:
                        pred_age = 'Old Age'

                if 'female' in texts[id]:
                    target_gender = 'female'
                else:
                    target_gender = 'male'

                if 'Infancy' in texts[id] or 'infancy' in texts[id]:
                    target_age = 'Infancy'
                elif 'Childhood' in texts[id] or 'childhood' in texts[id] or 'child' in texts[id] or 'Child' in texts[id]:
                    target_age = 'Childhood'
                elif 'Adolescence' in texts[id] or 'adolescence' in texts[id] or 'adolescent' in texts[id] or 'Adolescent' in texts[id]:
                    target_age = 'Adolescence'
                elif 'Young Adulthood' in texts[id] or 'young adulthood' in texts[id] or 'young adult' in texts[id] or 'Young Adult' in texts[id]:
                    target_age = 'Young Adulthood'
                elif 'Middle Age' in texts[id] or 'middle age' in texts[id] or self.other_middle in texts[id]:
                    target_age = 'Middle Age'
                elif 'Old Age' in texts[id] or 'old age' in texts[id]:
                    target_age = 'Old Age'
                else:
                    continue
                
                bias_list = explanation[people[0]]['bias']
                if 'gender' not in bias_list:
                    target_gender = None
                    pred_gender = None
                if 'age' not in bias_list:
                    target_age = None
                    pred_age = None

                if target_age == 'Middle Age':
                    print("middle_age_list ", self.middle_age_list," idx:", id, " if init_flag:", init_flag, " pred_gender:", pred_gender)

                if pred_gender == target_gender and pred_age == target_age:
                    continue
                else:
                    raw_image = faceedit.forward(img_path = img_save_path, gender = target_gender, age = target_age)

                    if id in self.middle_age_list and init_flag:

                        raw_image.save(f'/home/ubuntu/shengyin/ethical-lens/middle_age_test/rubbish_{id}.jpg')
                        tmp_data = {'image': {'path': f'/home/ubuntu/shengyin/ethical-lens/middle_age_test/rubbish_{id}.jpg'}}
                        middle_age_flag = False
                        
                        fairface_result = fairface.calculate(tmp_data)
                        if fairface_result['pred']['people'] != 1:
                            middle_age_flag = True
                        else:
                            for key in fairface_result['pred']['age']:
                                value = fairface_result['pred']['age'][key]
                                if value == 1:
                                    pred_age = key
                                    break    
                            if pred_age != '40-49':
                                middle_age_flag = True
                        
                        if middle_age_flag:

                            revision_text = texts[id].replace('middle age', self.other_middle)
                            if label == None:
                                label = 'K1'
                            tmp_label = label
                            tmp_data = {
                                'bias_pre_response': {
                                    'people': tmp_people,
                                    'explanation': tmp_explanation
                                }
                            }
                            new_middle_img, new_img_dict, _ = self.tox_post_w_bias([revision_text], tmp_label, tmp_data)
                            new_raw_image, new_img_dict = new_middle_img[0], new_img_dict['img_0']
                            print("Already make second image generated")

                            new_raw_images = self.face_edit([revision_text], [new_raw_image], tmp_people, tmp_explanation, label = tmp_label, init_flag = False, second_id = id)
                            print("Already make second image edited")
                            raw_image = new_raw_images[0]
                            raw_image.save(f'./results/tmp/_second_middle_age_faceedit_{id}.jpg')
                    
                    raw_images[id] = raw_image
                    continue
        
        return raw_images
    
    def regenerate_middle_age(self, raw_image, id, texts, label, img_dict, tmp_people, tmp_explanation):

        raw_image.save(f'/home/ubuntu/shengyin/ethical-lens/middle_age_test/rubbish_{id}.jpg')
        tmp_data = {'image': {'path': f'/home/ubuntu/shengyin/ethical-lens/middle_age_test/rubbish_{id}.jpg'}}
        middle_age_flag = False
        
        fairface_result = self.fairface.calculate(tmp_data)
        import pdb; pdb.set_trace()
        if fairface_result['pred']['people'] != 1:
            middle_age_flag = True
        else:
            for key in fairface_result['pred']['age']:
                value = fairface_result['pred']['age'][key]
                if value == 1:
                    pred_age = key
                    break    
            if pred_age != '40-49':
                middle_age_flag = True

        middle_age_flag = True
        if middle_age_flag:

            revision_text = texts[id].replace('middle age', self.other_middle)
            tmp_label = label[id]
            tmp_data = {}

            new_middle_img, new_img_dict = self.tox_post_w_bias(revision_text, tmp_label, tmp_data)
            raw_image, new_img_dict = new_middle_img[0], new_img_dict['img_0']
            img_dict[f'img_{id}'] = new_img_dict

            raw_images = self.face_edit([revision_text], [raw_image], tmp_people, tmp_explanation, tmp_label, img_dict, False)
            raw_image = raw_images[0]

            # face edit

        return raw_image

    def generate(self, prompt, img_number, data, state):
        if state == 'Yes':
            # toxicity pre-process
            text, label = self.tox_pre(prompt, img_number, data)
            print(text)
            if data['status'] == 'FAIL':
                return
    
            # bias pre-process
            revision, people, explanation = self.bias_pre(text, img_number, data)
            print(revision)
            print(people)
    
            # toxicity post-process
            if people == []:
                raw_images, img_dict, publics = self.tox_post_wo_bias(text, img_number, label, data)
            else:
                raw_images, img_dict, publics = self.tox_post_w_bias(revision, label, data)
            
            # face edit
            if len(people) == 0:
                data['images'] = raw_images
                return
            else:
                raw_images = self.face_edit(revision, raw_images, people, explanation, label = label)

            data['image_log'] = img_dict
    
            data['images'] = raw_images
        else:
            raw_images = self.text2img(
                prompt=prompt,
                num_images_per_prompt=img_number,
                num_inference_steps=25,
                width=self.resolution,
                height=self.resolution,
                requires_safety_checker=False
            )['images']
            data['images'] = raw_images
        
        return

if __name__ == "__main__":
    cfg = {
        't2i' : "/dev/shm/data/shengyin/pretrain_model/models--dreamlike-art--dreamlike-diffusion-1.0/snapshots/9fb5a6463bf79d81152e715e8d2a8b988f96c790",
        'resolution' : 512,
        'relative_gpu_index' : 0,

        'faceedit_device': 'cuda:1',
        'fairface' : {
            'relative_gpu_index' : "1",
            'SAVE_DETECTED_AT': "results/detected_faces_generate"
        },
        'if_fairface' : True,

        'checkpoints': "/home/ubuntu/yuzhucai/ethicallens_/checkpoints/v3",
        'clipseg_path': "/home/ubuntu/DATA1/yuzhucai/prestrain_model/CIDAS--clipseg-rd64-refined"
    }
    random.seed()
    random_seed = random.randint(0, 1000000)
    prepare_seed(random_seed)
    print(random_seed)
    gen = Ethicallens(cfg=cfg)
    data = {
            'seed': random_seed,
            'status': 'SUCSESS',
        }
    gen.generate("a doctor", 1, data=data, state="Yes")
    for img in data['images']:
        img.save('/home/ubuntu/yuzhucai/ethicallens_/model/test.jpg')