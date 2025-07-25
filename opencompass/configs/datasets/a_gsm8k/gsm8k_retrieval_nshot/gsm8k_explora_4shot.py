from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]
# Qwen2.5-7B
round = [
{'role': 'HUMAN', 'prompt': "Question: Pulsar, the shuffle-dancing bear, Polly, the pink prancing poodle, and Petra, the proud portly pachyderm, are entertainers at the Big Top Circus. In one show, Pulsar stands on his two back legs for a total of 10 minutes. Then, Polly stands on her back legs for three times as long as Pulsar. And then, finally, Petra stands on his back legs for one-sixth as long as Polly. What is the combined length of time, in minutes, that the three entertainers stand on their back legs?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "Pulsar stands on his back legs for 10 minutes.\nPolly stands for 3 times that, so 10 × 3 = 30 minutes.\nPetra stands for one-sixth as long as Polly, so 30 ÷ 6 = 5 minutes.\nIn total, they stand for 10 + 30 + 5 = 45 minutes.\nSo the answer is $\\boxed{45}$.\n"},
{'role': 'HUMAN', 'prompt': "Question: The cost of a slice of cake is three-fourths of the cost of a cup of milk tea. If the milk tea costs $2.40, how much do 2 slices of cake and 1 cup of milk tea cost?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "The cost of a slice of cake is 3/4 × $2.40 = $1.80.\nTwo slices of cake cost 2 × $1.80 = $3.60.\nAdding the cost of 1 cup of milk tea: $3.60 + $2.40 = $6.00.\nSo the answer is $\\boxed{6}$.\n"},
{'role': 'HUMAN', 'prompt': "Question: The bowling alley has 30 red bowling balls. There are 6 more green bowling balls than red. How many bowling balls are there in total?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "There are 30 red bowling balls.\nThere are 6 more green balls than red, so 30 + 6 = 36 green balls.\nTotal number of balls is 30 + 36 = 66.\nSo the answer is $\\boxed{66}$.\n"},
{'role': 'HUMAN', 'prompt': "Question: The chances of making the junior high basketball team start at 10% if you're 66 inches and increase 10% for every additional inch of height. Devin starts out as 65 inches tall, then grows 3 inches. What are his chances of making the basketball team?\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
{'role': 'BOT', 'prompt': "Devin's original height is 65 inches. He grows 3 inches, so his new height is 65 + 3 = 68 inches.\nCompared to 66 inches, Devin is now 2 inches taller.\nEach inch increases the chance by 10%, so 2 × 10% = 20% increase.\nBase chance is 10%, so final chance is 10% + 20% = 30%.\nSo the answer is $\\boxed{30\\%}$.\n"},
{'role': 'HUMAN', 'prompt': "Question: {question}\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer:"},
]


gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=round)),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096, stopping_criteria=stop_words))

gsm8k_eval_cfg = dict(evaluator=dict(type=Gsm8kEvaluator),
                      pred_postprocessor=dict(type=gsm8k_postprocess),
                      dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

gsm8k_datasets = [
    dict(
        abbr='gsm8k_explora_4shot',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
