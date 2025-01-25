import re
from tqdm import tqdm
import numpy as np

import torch

from huggingface_hub import login
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelHandler:
    def __init__(self, model_name: str, device:str, user_token=None):
        self.model_name = model_name
        self.model = None
        self.device = device
        self.tokenizer = None
        self.load_model(user_token)
        
    @staticmethod
    def check_citations(row, list_of_citations_in_answer):
        gt_pattern = r'\[\d+\]'
        text = row['rag_raw_answer']
        temp_matches = re.findall(gt_pattern, text)  
        matches = [item[1:-1] for item in set(temp_matches)]
        row['rag_citation'] = matches
        if set(matches) == set(list_of_citations_in_answer):
            return True, matches, list_of_citations_in_answer
        else:
            return False, matches, list_of_citations_in_answer
    
    @staticmethod
    def find_sequences(lst):
        indices = []
        for i in range(len(lst) - 2):
            if (lst[i] == '[' or  lst[i] == '][') and (lst[i+2] == ']' or lst[i+2] == '][' )and lst[i+1].isdigit():
                indices.append(i + 1)
        return indices
                
    def load_model(self, user_token):
        try:
            # Try to load the tokenizer and model
            print(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=user_token)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device, # or "auto"
            )
            print(f"Model {self.model_name} loaded successfully.")
        
        except RepositoryNotFoundError:
            # Handle case where the model is not found
            raise ValueError(f"Model {self.model_name} not found on Hugging Face Hub.")
        
        except HfHubHTTPError as e:
            # If the model requires login (401 error)
            if e.response.status_code == 401:
                print(f"Model {self.model_name} requires Hugging Face login.")
                
                if user_token is None:
                    raise ValueError(f"The model {self.model_name} is private. Please provide a Hugging Face token.")
                
                # Authenticate using the provided token
                print("Authenticating with Hugging Face...")
                login(token=user_token)
                print("Authenticated successfully. Trying to load the model again...")

                # Retry loading the tokenizer and model after authentication
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=user_token)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device, # or "auto"
                )
                print(f"Model {self.model_name} loaded successfully after authentication.")
            else:
                raise e  # Re-raise other HTTP errors

    def generate_with_probs(self, patt_input_data, prompt_template_system, prompt_template_user):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer need to be loaded first. Call `load_model()`.")
            
        pattern = r'[^\w\s\[\]]'
            
        nlg_with_responses = []
        for row in tqdm(patt_input_data):
            
            messages = [
                {"role": "system", "content": prompt_template_system},
                {"role": "user", "content": prompt_template_user.format(query=row['query'], search_results=row['doc_list_with_text'])},
            ]
    
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=256,
                return_dict_in_generate=True,
                output_scores = True,
                eos_token_id=terminators,
                do_sample=False,
                temperature=0.01,
                top_p=1,
            )

            transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
            )
    
            transition_scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
            
            generated_tokens = outputs[0][0][input_ids.shape[-1]:] #outputs.sequences[0]

            # prob
            decoded_token_list = []
            logprob_list = []
            for i, token_id in enumerate(generated_tokens):
                token = self.tokenizer.decode([token_id])
                cleaned_token = re.sub(pattern, '', token.replace('\n',' ').replace('\t', ' ').replace(' ',''))
                decoded_token_list.append(cleaned_token)
                logprob_list.append(transition_scores[0, i].item())
            
            citation_indices = self.find_sequences(decoded_token_list)
            
            logprob_list = np.array(logprob_list)
            decoded_token_list = np.array(decoded_token_list)
            list_of_citations_in_answer = decoded_token_list[citation_indices]
    
            logprob_of_citations_in_answer = logprob_list[citation_indices]
            # end of prob
            
            response = outputs[0][0][input_ids.shape[-1]:]
            decoded_response = self.tokenizer.decode(response, skip_special_tokens=True)
    
            row['rag_raw_answer'] = decoded_response
            check_citation_res = self.check_citations(row, list_of_citations_in_answer)
            if check_citation_res[0]==False:
                print("Citation Check failed!")
                print(check_citation_res[1])
                print(check_citation_res[2])
                # break

            row['list_of_citations_in_answer'] = [int(num) for num in list_of_citations_in_answer]
            row['logprob_of_citations_in_answer'] = logprob_of_citations_in_answer

            del generated_tokens
            del outputs
            nlg_with_responses.append(row)
        return nlg_with_responses


    

