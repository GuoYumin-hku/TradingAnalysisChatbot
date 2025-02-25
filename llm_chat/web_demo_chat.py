import os
import gradio as gr
from gradio.components import HTML
import torch
from PIL.ImageOps import scale
from PIL import Image
from transformers import BertTokenizer, BertForSequenceClassification
import json
from zhipuai import ZhipuAI

label_num_to_text = {0: ('Furniture', 'Bookcases'), 1: ('Furniture', 'Chairs'), 2: ('Furniture', 'Furnishings'), 3: ('Furniture', 'Tables'), 4: ('Office Supplies', 'Appliances'), 5: ('Office Supplies', 'Art'), 6: ('Office Supplies', 'Binders'), 7: ('Office Supplies', 'Envelopes'), 8: ('Office Supplies', 'Fasteners'), 9: ('Office Supplies', 'Labels'), 10: ('Office Supplies', 'Paper'), 11: ('Office Supplies', 'Storage'), 12: ('Office Supplies', 'Supplies'), 13: ('Technology', 'Accessories'), 14: ('Technology', 'Copiers'), 15: ('Technology', 'Machines'), 16: ('Technology', 'Phones')}

# Load the BERT model and tokenizer
model_path = '../Classification/results/checkpoint-13464'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(model_path)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Initialize ZhipuAI client
client = ZhipuAI(api_key="e2a004d7071b4fcabf1859496ad76124.Ej8iFCuCo1ghwpna")

# Function to classify commodity
def classify_commodity(commodity_name):
    inputs = tokenizer(commodity_name, return_tensors="pt").to(device)
    outputs = model(**inputs)
    preds = outputs.logits.argmax(-1).item()
    category = label_num_to_text[preds][0] + "-" + label_num_to_text[preds][1]
    print(category)
    return category

# Function to generate sales suggestions
def generate_suggestions(messages, commodity_name, category):

    analysis_file = f'../data_analysis/analysis_result_{category}/analysis_result_{category}.json'
    with open(analysis_file, 'r') as f:
        analysis_result = json.load(f)

    known_facts = {
        "product": category,
        "most_segment_customers": analysis_result["segment_distribution"],
        "most_quantity_customers": analysis_result["segment_quantity"],
        "highest_unit_price_analysis": analysis_result["segment_unit_price"],
        "highest_profit_analysis": analysis_result["segment_profit"],
        "highest_discount_analysis": analysis_result["segment_discount"],
        "most_country_customers": analysis_result["country_distribution"],
    }
    top_10_cities_image = Image.open(f'../data_analysis/analysis_result_{category}/top_10_cities_distribution_{category}.png')
    top_10_states_image = Image.open(f'../data_analysis/analysis_result_{category}/top_10_states_distribution_{category}.png')

    response = client.chat.completions.create(
        model="glm-zero-preview",
        messages=[
            {"role": "user", "content": f"As a Sales Assistant to help the company, please analyze and generate sales suggestions and advice for the product {category}, with less than 300 words."},
            {"role": "assistant", "content": "Of course, I can help you with that. Could you provide me with some information about the product?"},
            {"role": "user", "content": f"Here are the facts: {known_facts}"}
        ],
    )
    sales_suggestions = response.choices[0].message.content

    messages.append((commodity_name, f"The commodity '{commodity_name}' belongs to category '{category}'.\n\nSales Suggestions:\n{sales_suggestions}"))
    return messages, known_facts, top_10_states_image, top_10_cities_image

# Function to send a message to the chatbot to keep talking (remember using the history)
def send_message(chat_history, message):
    messages = []
    print(chat_history)
    for i in range(0, len(chat_history)):
        messages.append({
            "role": "user",
            "content": chat_history[i][0]
        })
        messages.append({
            "role": "assistant",
            "content": chat_history[i][1]
        })
    messages.append({
        "role": "user",
        "content": message
    })
    # using history to keep the conversation
    print(messages)
    response = client.chat.completions.create(
        model="glm-4-plus",
        messages=messages,
    )
    messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })
    chat_history.append([message,response.choices[0].message.content])
    return chat_history


# Create Gradio Blocks interface
with gr.Blocks() as demo:
    title = "# Sales Assistant Chatbot"
    gr.Markdown(title)
    category = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            commodity_input = gr.Textbox(label="Input Commodity Name")
            classify_button = gr.Button("Step1.Classify")
            category_output = gr.Textbox(label="Category", interactive=False)
            suggestions_button = gr.Button("Step2.Generate Suggestions")
            known_facts_output = gr.Textbox(label="Known Facts", interactive=False)
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="Sales Assistant Chatbot")
            chatbot_input = gr.Textbox(label="Your Message")
            send_button = gr.Button("Continue Chat")
            with gr.Row():
                # 2 images in a row
                image1 = gr.Image(value=None, label="Top 10 States Distribution")
                image2 = gr.Image(value=None, label="Top 10 Cities Distribution")



    classify_button.click(fn=classify_commodity, inputs=commodity_input, outputs=category_output)
    # suggestions_button.click(fn=generate_suggestions, inputs=[chatbot, commodity_input, category_output], outputs=[chatbot, known_facts_output, image1, image2])
    suggestions_button.click(
        fn=lambda chatbot, commodity_name, category: (
            *generate_suggestions(chatbot, commodity_name, category),
        ),
        inputs=[chatbot, commodity_input, category_output],
        outputs=[chatbot, known_facts_output, image1, image2],
    )
    send_button.click(fn=send_message, inputs=[chatbot, chatbot_input], outputs=chatbot)

demo.launch(share=True)