openai_api_key = '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'



from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import streamlit as st
import magic
from PIL import Image



st.set_page_config(page_title="MPR Project group no ", page_icon=":robot:")

header_style = """
    <style>
        .stApp header {
            background-color: #f0f0f0;
            padding: 20px;
        }
    </style>
"""

h1_style = """
    <style>
        .stApp h1 {
            color: #333;
            font-size: 36px;
            font-weight: bold;
        }
    </style>
"""

text_input_style = """
    <style>
        .stTextInput>div>div>div>input {
            border: 2px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
        }
    </style>
"""

button_style = """
    <style>
        .stButton>button {
            background-color: #007bff;
            color: #fff;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        
        .stButton>button:hover {
            background-color: #0056b3;
        }
    </style>
"""

# Display inline CSS styles
st.markdown(header_style, unsafe_allow_html=True)
st.markdown(h1_style, unsafe_allow_html=True)
st.markdown(text_input_style, unsafe_allow_html=True)
st.markdown(button_style, unsafe_allow_html=True)
# st.image("C:\\Users\\Admin\\Desktop\\MPRsalesnmarketing\\logo.jpeg")
img1 = Image.open('logo2.jpeg')
img1 = img1.resize((156,145))
st.image(img1,use_column_width=False)
# st.header("RAPID ROAD'S HELPER")
st.header("CyberPitch: AI-Driven Business Proposal Generation from Company Insights")

st.write("Enter the company name to whom you want send the email to:")
def get_companytext():
    input_text = st.text_area(label="Company name", label_visibility='collapsed', placeholder="enter Company name", key="")
    return input_text

company_input= get_companytext()



def get_company_page(company_path):
    y_combinator_url = f"https://www.ycombinator.com/companies/{company_path}"
    
    # print (y_combinator_url)

    loader = UnstructuredURLLoader(urls=[y_combinator_url])
    return loader.load()


# Get the data of the company you're interested in
data = get_company_page(company_input)
    
# print (f"You have {len(data)} document(s)")

# print(data)
# print (f"Preview of your data:\n\n{data[0].page_content[:30]}")

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 800,
    chunk_overlap  = 0
)

docs = text_splitter.split_documents(data)

# print (f"You now have {len(docs)} documents")

map_prompt = """Below is a section of a website about {prospect}

Write a concise summary about {prospect}. If the information is not about {prospect}, exclude it from your summary.

{text}

% CONCISE SUMMARY:"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "prospect"])

combine_prompt = """
Your goal is to write a personalized outbound email from {sales_rep}, a sales rep at {company} to {prospect}.

A good email is personalized and combines information about the two companies on how they can help each other.
Be sure to use value selling: A sales methodology that focuses on how your product or service will provide value to the customer instead of focusing on price or solution.

% INFORMATION ABOUT {company}:
{company_information}

% INFORMATION ABOUT {prospect}:
{text}

% INCLUDE THE FOLLOWING PIECES IN YOUR RESPONSE:
- Start the email with the sentence: "We love that {prospect} helps teams..." then insert what they help teams do.
- The sentence: "We can help you do XYZ by ABC" Replace XYZ with what {prospect} does and ABC with what {company} does 
- A 1-2 sentence description about {company}, be brief
- End your email with a call-to-action such as asking them to set up time to talk more
- Sign off the email with thankyou and {sales_rep}

% YOUR RESPONSE:
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["sales_rep", "company", "prospect", \
                                                                         "text", "company_information"])

# * RapidRoad helps product teams build product faster
# * We have a platform that allows product teams to talk more, exchange ideas, and listen to more customers
company_information = """
* Automated project tracking: 
RapidRoad could use machine learning algorithms to automatically track project progress, identify potential bottlenecks, and suggest ways to optimize workflows. This could help product teams stay on track and deliver faster results.
* Collaboration tools: RapidRoad could offer built-in collaboration tools, such as shared task lists, real-time messaging, and team calendars. This would make it easier for teams to communicate and work together, even if they are in different locations or time zones.
* Agile methodology support: RapidRoad could be specifically designed to support agile development methodologies, such as Scrum or Kanban. This could include features like sprint planning, backlog management, and burndown charts, which would help teams stay organized and focused on their goals.
"""

st.write("Enter the employee name :")
def get_ownname():
    input_text = st.text_area(label="oname", label_visibility='collapsed', placeholder="enter your name", key="1")
    return input_text
username= get_ownname()

# st.write("enter your company name :")
# def get_ownercompanyname():
#     input_text = st.text_area(label="cname", label_visibility='collapsed', placeholder="enter your comapny name", key="2")
#     return input_text
# usercompany= get_ownercompanyname()

st.write("Enter name of the prospect (actual name of the company):")
def get_prospect():
    input_text = st.text_area(label="prospect", label_visibility='collapsed', placeholder="enter your prospect", key="3")
    return input_text
prospectname= get_prospect()

if prospectname:

    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    chain = load_summarize_chain(llm,
                                chain_type="map_reduce",
                                map_prompt=map_prompt_template,
                                combine_prompt=combine_prompt_template,
                                verbose=True
                                )

    output = chain({"input_documents": docs, # The seven docs that were created before
                    "company": "Rapid Road", 
                    "company_information" : company_information,
                    "sales_rep" : username, 
                    "prospect" : prospectname
                })

    
   

    st.markdown("**your generated email is as follows:**")
    st.write(output['output_text'])

    print (output['output_text'])
# ----------------------------------------------------------------------------------------------------
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import magic
import os
import nltk
import pdfplumber
from io import StringIO 

st.markdown("---------------------------------------------------------------")
st.header("DocInquire: Ask Anything from Uploaded PDFs")
st.write("please upload the pdf file ")

import streamlit as st


# uploaded_file = st.file_uploader("Add text file !")
# if uploaded_file:
#     st.write("uploaded file successfully")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.write("uploaded file successfully")
#     # # To read file as bytes:
#     # bytes_data = uploaded_file.getvalue()
#     # st.write(bytes_data)

#     # # To convert to a string based IO:
#     # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
#     # st.write(stringio)

#     # # To read file as string:
#     # string_data = stringio.read()
#     # st.write(string_data)

#     # # Can be used wherever a "file-like" object is accepted:
#     # dataframe = pd.read_csv(uploaded_file)
#     # st.write(dataframe)
#     st.success("Uploaded the file")
    with pdfplumber.open(uploaded_file) as file:
        all_pages = file.pages
        # st.write(all_pages[0].readlines())
# print the content of the uploaded file 
        st.write(all_pages[0].extract_text())


        # faiss_index = FAISS.from_documents(all_pages, OpenAIEmbeddings(openai_api_key=openai_api_key))
        # docs = faiss_index.similarity_search("does rapid road support agile development?", k=2)
        # for doc in docs:
        #     st.write(str(doc.metadata["page"]) + ":", doc.page_content[:300])

        # Load up your text into documents
        # documents = all_pages[0]
        documents=all_pages
        # Get your text splitter ready
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # Split your documents into texts
        splits = text_splitter.split_text(all_pages[0].extract_text())
        texts = text_splitter.create_documents(splits)
        # Turn your texts into embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Get your docsearch ready
        docsearch = FAISS.from_documents(texts, embeddings)
        # Load up your LLM
        llm = OpenAI(openai_api_key=openai_api_key)
        # Create your Retriever
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
        # Run a query

        st.write("Enter the question you would like to get an answer to based on the pdf you uploaded here::")
        def get_queryans():
            input_text = st.text_area(label="qame", label_visibility='collapsed', placeholder="enter your query")
            return input_text
        query= get_queryans()

        # query = "should i use rapidroad for my agile project needs? what such agile technologies does it use?"
        # qa.run(query)
        if len(query)>0:
            st.write("ANS:")
            st.write(qa.run(query))
# from pathlib import Path
# loader = DirectoryLoader(Path("C:/Users/Admin/Desktop/MPRsalesnmarketing/"), glob='**/*.txt')



