# LangChain分析 - Memory

## 1. 关键特性

默认情况下，Chains和Agents是无状态的，这意味着它们对待每个传入的查询都是独立的（就像底层的LLM和聊天模型本身一样）。在某些应用中，比如聊天机器人，记住之前的交互是至关重要的，无论是短期还是长期记忆。Memory类正是为此而设计。

LangChain以两种形式提供内存组件。

首先，LangChain提供了管理和操作先前聊天消息的辅助工具。这些工具被设计成模块化的，无论如何使用它们，都非常实用。

其次，LangChain提供了将这些工具轻松整合到Chains中的方法。

内存（Memory）是在用户与语言模型交互过程中保持状态的概念。用户的交互通过ChatMessages（聊天消息）来捕获。

内存涉及从一系列聊天消息中摄取、捕获、转换和提取知识的过程。

LangChain提供了不同类型的内存，每种类型有两种方式来理解和使用内存

。首先，可以使用独立的功能从一系列消息中提取信息。

其次，可以将这些内存类型用于Chains，用于在Chains中保留状态和信息。内存可以返回多个信息，例如最近的N条消息和所有先前消息的摘要，返回的信息可以是字符串或消息列表。

### 1.1 ChatMessageHistory

这是一个超轻量级的包装器，它提供了方便的方法来保存人类消息、AI消息，然后获取它们所有的信息。

## 1.2 ConversationBufferMemory

首先，我们展示ConversationBufferMemory，它只是ChatMessageHistory的一个包装器，用于从一个变量中提取消息。

## 1.3 在chain中使用ConversationBufferMemory。
```python
from langchain.llms import OpenAI
from langchain.chains import ConversationChain


llm = OpenAI(temperature=0)
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)
conversation.predict(input="Hi there!")
```

打印出的prompt 如下：

```shell script


    > Entering new ConversationChain chain...
    Prompt after formatting:
    The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

    Current conversation:

    Human: Hi there!
    AI:

    > Finished chain.





    " Hi there! It's nice to meet you. How can I help you today?"
```

```python
conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
```

打印出的prompt 如下：

```shell script


    > Entering new ConversationChain chain...
    Prompt after formatting:
    The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

    Current conversation:
    Human: Hi there!
    AI:  Hi there! It's nice to meet you. How can I help you today?
    Human: I'm doing well! Just having a conversation with an AI.
    AI:

    > Finished chain.





    " That's great! It's always nice to have a conversation with someone new. What would you like to talk about?"
````
## 1.4 保存Message History

Message History 可以导出为字典。

```python
import json

from langchain.memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict

history = ChatMessageHistory()

history.add_user_message("hi!")

history.add_ai_message("whats up?")

dicts = messages_to_dict(history.messages)

```


## 2. 代码模块

```
memory/
    ├── chat_message_histories/         # ChatMessageHistory模块
    │   ├── in_memory.py                # 定义ChatMessageHistory类，存储在内存中
    │   ├── cassandra.py                # 把ChatMessageHistory存储到cassandra数据库中
    │   ├── cosmos_db.py                # 把ChatMessageHistory存储到cosmos db中
    │   ├── ......                      # 各种存储ChatMessageHistory的方法
    ├── buffer.py                       # ConversationBufferMemory类，存储conversation memory的 buffer，
    ├── summary.py                      # 对conversation memory进行汇总
    ├── summary_buffer.py               # Buffer with summarizer for storing conversation memory
    ├── vectorstore.py                  # Class for a VectorStore-backed memory object
    ├── ......
    
```

## 3. 粗粒度代码逻辑
