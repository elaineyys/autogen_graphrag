import autogen
from rich import print
import chainlit as cl
from typing_extensions import Annotated
from chainlit.input_widget import (
   Select, Slider, Switch)
from autogen import AssistantAgent, UserProxyAgent
from utils.chainlit_agents import ChainlitUserProxyAgent, ChainlitAssistantAgent
# from graphrag.query.cli import run_global_search, run_local_search
from graphrag.cli.query import run_global_search, run_local_search

# LLama3 LLM from Lite-LLM Server for Agents #
llm_config_autogen = {
    "seed": 42,  # change the seed for different trials
    "temperature": 0,
    "config_list": [{"model": "litellm", 
                     "base_url": "http://0.0.0.0:4000/", 
                     'api_key': 'ollama'},
    ],
    "timeout": 60000,
}

@cl.on_chat_start
async def on_chat_start():
  try:
    settings = await cl.ChatSettings(
            [      
                Switch(id="Search_type", label="(GraphRAG) Local Search", initial=True),       
                Select(
                    id="Gen_type",
                    label="(GraphRAG) Content Type",
                    values=["prioritized list", "single paragraph", "multiple paragraphs", "multiple-page report"],
                    initial_index=1,
                ),          
                Slider(
                    id="Community",
                    label="(GraphRAG) Community Level",
                    initial=0,
                    min=0,
                    max=2,
                    step=1,
                ),

            ]
        ).send()

    response_type = settings["Gen_type"]
    community = settings["Community"]
    local_search = settings["Search_type"]
    
    cl.user_session.set("Gen_type", response_type)
    cl.user_session.set("Community", community)
    cl.user_session.set("Search_type", local_search)

    # retriever   = AssistantAgent(
    #    name="Retriever", 
    #    llm_config=llm_config_autogen, 
    #    system_message="""Only execute the function query_graphRAG to look for context. 
    #                 Output 'TERMINATE' when an answer has been provided.""",
    #    max_consecutive_auto_reply=1,
    #    human_input_mode="NEVER", 
    #    description="Retriever Agent"
    # )

    user_proxy = ChainlitUserProxyAgent(
        name="User_Proxy",
        human_input_mode="ALWAYS",
        llm_config=llm_config_autogen,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config=False,
        system_message='''A human admin.''',
        description="User Proxy Agent"
    )

    predictor = AssistantAgent(
        name="Predictor",
        is_termination_msg=lambda x: "TERMINATE" in str(x.get("content", "")).upper(),
        system_message="You are a medical expert specializing in CHF prediction.",
        llm_config=llm_config_autogen,
        description="Medical expert predicting CHF within 5 years.",
    )

    critic = AssistantAgent(
        name="Critic",
        is_termination_msg=lambda x: "TERMINATE" in str(x.get("content", "")).upper(),
        system_message="You are an assistant evaluating and providing feedback for CHF predictions.",
        llm_config=llm_config_autogen,
        description="Critic providing feedback for predictions.",
    )

    assistant = autogen.AssistantAgent(
        name="Assistant",
        human_input_mode="NEVER",
        system_message="""Only execute the function to retrieve for context.""",
        llm_config=llm_config_autogen
    )

    helper = autogen.AssistantAgent(
        name="Helper",
        human_input_mode="NEVER",
        system_message="""Interact with the retriever to provide any context.""",
        llm_config=llm_config_autogen
    )
    
    print("Set agents.")

    # cl.user_session.set("Query Agent", user_proxy)
    # cl.user_session.set("Retriever", retriever)

    cl.user_session.set("User Proxy", user_proxy)
    cl.user_session.set("Predictor", predictor)
    cl.user_session.set("Critic", critic)
    # cl.user_session.set("Assistant", assistant)
    cl.user_session.set("Helper", helper)

    msg = cl.Message(content=f"""Hello! What task would you like to get done today?      
                     """, 
                     author="User_Proxy")
    await msg.send()

    print("Message sent.")
    
  except Exception as e:
    print("Error: ", e)
    pass

@cl.on_settings_update
async def setup_agent(settings):
    response_type = settings["Gen_type"]
    community = settings["Community"]
    local_search = settings["Search_type"]
    cl.user_session.set("Gen_type", response_type)
    cl.user_session.set("Community", community)
    cl.user_session.set("Search_type", local_search)
    print("on_settings_update", settings)

@cl.on_message
async def run_conversation(message: cl.Message):
    print("Running conversation")
    INPUT_DIR = None
    ROOT_DIR = '.'    
    CONTEXT = message.content
    MAX_ITER = 10   
    RESPONSE_TYPE = cl.user_session.get("Gen_type")
    COMMUNITY = cl.user_session.get("Community")
    LOCAL_SEARCH = cl.user_session.get("Search_type")

    # retriever   = cl.user_session.get("Retriever")
    # user_proxy  = cl.user_session.get("Query Agent")

    predictor = cl.user_session.get("Predictor")
    critic = cl.user_session.get("Critic")
    user_proxy = cl.user_session.get("User Proxy")
    # assistant = cl.user_session.get("Assistant")
    helper = cl.user_session.get("Helper")

    print("Setting groupchat")

    user_proxy.reset()
    predictor.reset()
    critic.reset()
    helper.reset()
    # assistant.reset()

    # def state_transition(last_speaker, groupchat):
    #     messages = groupchat.messages
    #     if last_speaker is user_proxy:
    #         return retriever
    #     if last_speaker is retriever:
    #         if messages[-1]["content"].lower() not in ['math_expert','physics_expert']:
    #             return user_proxy
    #         else:
    #             if messages[-1]["content"].lower() == 'math_expert':
    #                 return user_proxy
    #             else:
    #                 return user_proxy
    #     else:
    #         pass
    #         return None
        
    previous = {"value": None}
    def state_transition(last_speaker, groupchat):
        messages = groupchat.messages
        print("previous:", previous["value"])
        if last_speaker is user_proxy:
            previous["value"] = 'user_proxy'
            return helper
        elif last_speaker is helper:
        #     return assistant
        # if last_speaker is assistant:
            # print(last_speaker, messages[-1]["content"].lower())
            if previous['value'] == 'user_proxy':
                previous['value'] = 'helper'
                print('1')
                return predictor
            else: #previous['value'] == 'predictor'
                previous['value'] = 'helper'
                return critic
        elif last_speaker is predictor:
            if previous['value'] not in ['critic']: # == 'helper'
                previous['value'] = 'predictor'
                # return assistant
                return helper
            else: #previous['value'] == 'critic'
                pass
        elif last_speaker is critic:
            previous['value'] = 'critic'
            return predictor
        else:
            print(last_speaker, previous['value'])
    
    async def query_graphRAG(
          question: Annotated[str, 'Query string containing information that you want from RAG search']
                          ) -> str:
        print(f"Invoking GraphRAG retrieval with question: {question}")
        if LOCAL_SEARCH:
            print(LOCAL_SEARCH)
            result = run_local_search(INPUT_DIR, ROOT_DIR, COMMUNITY ,RESPONSE_TYPE, question)
        else:
            result = run_global_search(INPUT_DIR, ROOT_DIR, COMMUNITY ,RESPONSE_TYPE, question)
        await cl.Message(content=result).send()
        return result

    # for caller in [retriever]:
    #     d_retrieve_content = caller.register_for_llm(
    #         description="retrieve content for code generation and question answering.", api_style="function"
    #     )(query_graphRAG)

    # for agents in [user_proxy, retriever]:
    #     print(agents)
    #     agents.register_for_execution()(d_retrieve_content)

    for caller in [helper]:
        d_retrieve_content = caller.register_for_llm(
            description="retrieve content for code generation and question answering.", api_style="function"
        )(query_graphRAG)

    # for agents in [helper, assistant]:
    #     agents.register_for_execution()(d_retrieve_content)
    for agents in [helper, predictor]:
        agents.register_for_execution()(d_retrieve_content)

#     groupchat = autogen.GroupChat(
#         agents=[user_proxy, retriever],
#         messages=[],
#         max_round=MAX_ITER,
#         # speaker_selection_method=state_transition,
#         allow_repeat_speaker=True,
#     )

    groupchat = autogen.GroupChat(
        # agents=[user_proxy, helper, assistant, predictor, critic],
        agents=[user_proxy, helper, predictor, critic],
        messages=[],
        max_round=MAX_ITER,
        speaker_selection_method=state_transition,
        allow_repeat_speaker=True,
    )
        
    manager = autogen.GroupChatManager(groupchat=groupchat,
                                       llm_config=llm_config_autogen, 
                                       is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
                                       code_execution_config=False,
                                       )    

# -------------------- Conversation Logic. Edit to change your first message based on the Task you want to get done. ----------------------------- # 
    if len(groupchat.messages) == 0: 
      await cl.make_async(user_proxy.initiate_chat)( manager, message=CONTEXT, )
    elif len(groupchat.messages) < MAX_ITER:
      await cl.make_async(user_proxy.send)( manager, message=CONTEXT, )
    elif len(groupchat.messages) == MAX_ITER:  
      await cl.make_async(user_proxy.send)( manager, message="exit", )