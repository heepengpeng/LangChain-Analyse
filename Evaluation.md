# Langchain 分析 - Evaluation

## 1. 关键特性

语言模型可能是不可预测的。这使得将可靠的应用程序部署到生产环境变得具有挑战性，因为在各种输入上产生可重复、有用的结果是一个最基本的要求。
测试有助于证明LLM应用程序中的每个组件能够产生所需或预期的功能。

LangChain提供了不同类型的评估器，用于常见类型的评估。每种类型都有现成的实现供您开始使用，并提供可扩展的API

以下是一些示例：

字符串评估器：评估给定输入的预测字符串，通常与参考字符串进行对比。
轨迹评估器：评估代理动作的整个轨迹。
比较评估器：对比在共同输入上的两次运行的预测结果。


### 1.1 字符串评估器

#### 1.1.1 自定义标准评估
使用 CriteriaEvalChain 根据自定义标准（包括自定义评分标准和宪法原则）评估模型输出。

```python
from langchain.evaluation import load_evaluator

evaluator = load_evaluator("criteria", criteria="conciseness")

# This is equivalent to loading using the enum
from langchain.evaluation import EvaluatorType

evaluator = load_evaluator(EvaluatorType.CRITERIA, criteria="conciseness")

eval_result = evaluator.evaluate_strings(
    prediction="What's 2+2? That's an elementary question. The answer you're looking for is that two and two is four.",
    input="What's 2+2?",
)
print(eval_result)
```

```shell script
    {'reasoning': 'The criterion is conciseness, which means the submission should be brief and to the point. \n\nLooking at the submission, the answer to the question "What\'s 2+2?" is indeed "four". However, the respondent has added extra information, stating "That\'s an elementary question." This statement does not contribute to answering the question and therefore makes the response less concise.\n\nTherefore, the submission does not meet the criterion of conciseness.\n\nN', 'value': 'N', 'score': 0}
```

#### 1.1.2 自定义字符串评估器

您可以通过继承 StringEvaluator 类并实现 _evaluate_strings（以及用于异步支持的 _aevaluate_strings）方法来创建自己的自定义字符串计算器。

#### 1.1.3 embedding距离

要测量预测和参考标签字符串之间的语义相似性（或相异性），您可以使用向量向量距离度量来使用 embedding_distance 评估器来衡量两个嵌入表示。

注意：这会返回一个距离分数，这意味着数字越低，根据embedding的表示，预测与参考越相似。

#### 1.1.4 QA Correctness

在考虑 QA 系统时，最重要的问题之一是最终生成的结果是否正确。 “qa”评估器将问答模型的响应与参考答案进行比较，以提供此级别的信息。 如果您能够注释测试数据集，则此评估器将很有用。

#### 1.1.5 String Distance

将 LLM 或链的字符串输出与参考标签进行比较的最简单方法之一是使用字符串距离测量，例如 Levenshtein 或后缀距离。 这可以与近似/模糊匹配标准一起用于非常基本的单元测试。


### 1.2 比较不同的Evaluators

LangChain 中的比较评估器有助于测量两个不同的链或 LLM 输出。 这些评估器有助于比较分析，例如两种语言模型之间的 A/B 测试，或比较同一模型的不同版本。 它们对于生成人工智能辅助强化学习的偏好分数等也很有用。

这些求值器继承自 PairwiseStringEvaluator 类，为两个字符串提供比较接口 - 通常是两个不同提示或模型的输出，或者同一模型的两个版本。 本质上，比较评估器对一对字符串执行评估，并返回包含评估分数和其他相关详细信息的字典。

要创建自定义比较计算器，请继承 PairwiseStringEvaluator 类并覆盖 _evaluate_string_pairs 方法。 如果需要异步计算，还需覆盖 _aevaluate_string_pairs 方法。

以下是比较评估器的关键方法和属性的摘要：

evaluate_string_pairs：评估输出字符串对。 创建自定义评估器时应覆盖此函数。
aevaluate_string_pairs：异步评估输出字符串对。 应重写此函数以进行异步计算。
require_input：此属性指示该求值器是否需要输入字符串。
require_reference：此属性指定此计算器是否需要参考标签。
以下部分提供了有关创建自定义评估器和可用的内置比较评估器的详细信息。

### 1.3 轨迹Evaluators

在LangChain中，轨迹评估器（Trajectory Evaluators）提供了一种更全面的方法来评估代理（agent）。这些评估器评估代理执行的完整操作序列及其相应的响应，我们将其称为“轨迹”（trajectory）。这使您能够更好地衡量代理的效果和能力。

轨迹评估器实现了AgentTrajectoryEvaluator接口，该接口需要两个主要方法：

evaluate_agent_trajectory：该方法同步评估代理的轨迹。
aevaluate_agent_trajectory：这个异步版本允许并行运行评估，以提高效率。
这两个方法都接受三个主要参数：

input：给定给代理的初始输入。
prediction：代理的最终预测响应。
agent_trajectory：代理执行的中间步骤，以元组列表的形式给出。
这些方法返回一个字典。建议自定义实现返回一个得分（表示代理效果的浮点数）和推理（解释得分背后的原因的字符串）。

您可以通过使用return_intermediate_steps=True参数来捕获代理的轨迹。这样可以收集所有的中间步骤，而无需依赖特殊的回调函数。

## 2. 代码模块

```
evaluation/                                        # 评估模块
    ├── agents/                                    # 评估Agent的轨迹
    │   ├── trajectory_eval_chain.py               # 定义比较ReAct类型agent的轨迹的chain
    ├── comparison/               
    │   ├── eval_chain.py                          # 定义比较两个模型输出的chain
    ├── criteria/                
    │   ├── eval_chain.py                          # 用于根据标准评估运行的LLM chain。    
    ├── embedding_distance/                
    │   ├── base.py                                # 根据embedding距离来进行评估的 chain   
    ├── qa/                       
    │   ├── eval_chain.py                          # 专门用于评估问答任务的LLM chain 
    ├── string_distance/                           
    │   ├── base.py                                # 根据string 的距离进行评估的chain
```

## 3. 核心代码逻辑

