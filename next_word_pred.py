import os
import string
import streamlit as st
from transformers import BertTokenizer, BertForMaskedLM
import torch
from torch.nn import functional as F


@st.cache(allow_output_mutation=True)
def load_model():
  try:
      bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
      return bert_tokenizer, bert_model
  except Exception as e:
    pass


def encode_text(tokenizer, seed_text, special_tokens = True):
  seed_text = seed_text.replace('<mask>', tokenizer.mask_token)

  #adding dummy puntuation after mask, to make sure model doesn't predict puntuation
  if tokenizer.mask_token == seed_text.split()[-1]:
        seed_text += ' .'
  input_ids = torch.tensor([tokenizer.encode(seed_text, add_special_tokens=special_tokens)])
  mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
  return input_ids, mask_idx


def decode_predictions(tokenizer, pred_idx, top_clean):
  ignore_tokens = string.punctuation + '[PAD]'
  
  tokens  = []

  for idx in pred_idx:
    token = ''.join(tokenizer.decode(idx).split())

    #to ignore puntuation and <PAD> tokens
    if token not in ignore_tokens:
          tokens.append(token.replace('##', ''))
  
  return '\n'.join(tokens[:top_clean])

def get_all_predictions(text_sentence,tokenizer,model, top_k=5):

  ip_enc, pred_idx = encode_text(tokenizer, text_sentence)
  
  #infering from the pre_trained_model
  with torch.no_grad():
    predict = model(ip_enc)[0]
  output = decode_predictions(tokenizer, predict[0, pred_idx, :].topk(20).indices.tolist(), top_k)
  return output

def get_prediction(input_text):
  try:
    input_text += ' <mask>'
    res = get_all_predictions(input_text, tokenizer, model, top_k=int(top_k))
    return res
  except Exception as error:
    pass


#============================interface=================================#
try:
  st.set_page_config(page_title='NextWordPredictor', layout = 'wide', initial_sidebar_state = 'auto')
  st.title("Next Word Prediction")

  input_text =  st.text_area('Enter your text here', placeholder='Type here....')
  top_k = st.slider("Pick a K, for top_k words for next word", 1, 8, 1)
 
  tokenizer, model = load_model()

  button = st.button(f'Predict the top {top_k} words')

  if button:
    if len(input_text) == 0:
      st.text('Please enter input text and try again!!')
    
    else:
      res = get_prediction(input_text)

      ans = []
      for i in res.split("\n"):
        ans.append(i)

      ans_as_str = "    ".join(ans)
      st.text(ans_as_str)

except Exception as e:
  pass
