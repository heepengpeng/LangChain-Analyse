# LangChain分析 - Agent

## 1. 关键特性

Agent的核心思想是用LLM选择一系列要执行的动作。在Chain中，这些动作被硬编码在代码中。  
而在agent中，使用LLM作为推理引擎，来确定要采取的动作及其顺序。


###  Agent
Agent 是负责下一步动作的类，由一个语言模型和提示词来驱动。这个提示词可以是agent的个性，agent的背景信息。

Agent分为两大类：Action Agent 和 Plan-and-execute agents。

#### Action Agent

Action Agent 有：
* Zero-shot ReAct。根据 tool的描述信息决定使用哪一个tool。
* Structured input ReAct。 能够使用多输入tool，之前的agent将输入固定为单个字符串。而这个代理可以使用tool的参数模式，创建结构化的动作输入。例如在浏览器中进行精确定位导航。
* OpenAI Functions。某些OpenAI模型已经进行了微调，以便检测何时应该调用函数，并回复硬干传递给函数的输入。
* Conversational。这个Agent专为对话场景设计。其提示被设计成使代理变得有益且易于对话。它使用 ReAct 框架来决定使用哪个工具，并利用内存来记住之前的对话交互。
* Self ask with search。
这个代理使用一个名为"Intermediate Answer"的工具。这个工具可以查找问题的事实答案。这个代理相当于最初的"自问自答与搜索论文"，其中提供了谷歌搜索API作为工具。
* ReAct document store。这个代理使用 ReAct 框架与文档存储进行交互。必须提供两个工具：一个搜索工具和一个查找工具（它们的名称必须完全一样）。搜索工具用于搜索文档，而查找工具用于在最近找到的文档中查找术语。

#### Plan-and-execute agents
 Plan-and-execute agents 通过首先规划要做的事情，然后执行子任务来实现目标。


### Tools

Tools 是Agent调用的函数。LangChain提供了一系列广泛的工具供您开始使用，同时还可以轻松定义您自己的工具（包括自定义描述）。

### Toolkits
这些是实现特定目标所需的工具组合。通常一个工具包中会包含大约3-5个工具。

### AgentExecutor
AgentExecutor是Agent的运行时环境。它实际上调用Agent并执行它选择的动作。以下是该运行时的伪代码：

```
next_action = agent.get_action(...)
while next_action != AgentFinish:
    observation = run(next_action)
    next_action = agent.get_action(..., next_action, observation)
return next_action
```

虽然这看起来很简单，但这个运行时实际上为您处理了几个复杂性，包括：

1. 处理代理选择不存在的工具的情况
2. 处理工具错误的情况
3. 处理代理产生的输出无法解析为工具调用的情况
4. 记录和监控所有层面的信息（包括代理决策和工具调用），可以输出到stdout或LangSmith。

## 2. 代码模块

```
agents/
    ├── agent.py                        # Agent的核心文件。Agent类，调用llm，决定下一步的动作。AgentExecutor，包含一个agent及其使用的tools，执行agent获取输出。
    ├── agent_types.py                  # Agent类型的枚举。包括ZERO_SHOT_REACT_DESCRIPTION,REACT_DOCSTORE,SELF_ASK_WITH_SEARCH,CONVERSATIONAL_REACT_DESCRIPTION等，共9种。
    ├── tools.py                        # 定义InvalidTool。Invalid Tool指的是当调用的tool的名字不合法时，调用这个tool。
    ├── agent_toolkits/                 # 定义各种ToolsKit。
    │   ├── csv                         # 处理csv文件的Toolkit。
    │   ├── gmail                       # 与Gmail交互的Toolkit。
    │   ├── python                      # 执行Python代码的Toolkit。
    │   ├── ......                      # Toolkit 可扩展
    ├── chat/                           # chat场景下的Zero-shot ReAct。
    ├── conversational/                 # 这个Agent专为对话场景设计。其提示被设计成使代理变得有益且易于对话。
    ├── conversational_chat/            # chat场景下的conversational
    ├── mrkl/                           # MRKL系统：一种模块化、神经符号结构的架构，将大型语言模型、外部知识源和离散推理相结合。
    ├── openai_functions_agent/         # 由OpenAI函数驱动的agent
    ├── openai_functions_multi_agent/   # 由OpenAI函数驱动的agent
    ├── react/                          # 实现了ReAct算法的Chain
    ├── self_ask_with_search            # 这个代理使用一个名为"Intermediate Answer"的工具。这个工具可以查找问题的事实答案
    └── structured_chat                 # 能够使用多输入的tool

tools/
    ├── base.py                         # Tools的核心实现逻辑
    ├── arxiv/                          # Tool for the Arxiv API
    ├── bing_search/                    # Bing Search API toolkit
    ├── ....../                         # 可扩展    
## 3. 粗粒度代码逻辑
