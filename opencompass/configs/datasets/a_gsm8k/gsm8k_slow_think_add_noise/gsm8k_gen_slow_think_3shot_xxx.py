from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","\n\nQuestion:","<|end_of_text|>"," Question","\n[Question]"]
demo = []
demo.append("""Okay, let's see. So, Kalinda and her mom are working on a 360-piece puzzle. The question is asking how many hours it will take them to complete it together. Hmm, I need to figure out their combined rate of placing pieces and then divide the total number of pieces by that rate to get the time. Then convert that time from minutes to hours. Let me break it down step by step.

First, Kalinda's rate. The problem says she can normally add 4 pieces per minute. So, Kalinda's rate is 4 pieces per minute. Then, her mom can place half as many pieces per minute as Kalinda. Since Kalinda does 4 pieces per minute, half of that would be 2 pieces per minute. So, her mom's rate is 2 pieces per minute.

Wait, let me make sure I read that correctly. "Her mom can typically place half as many pieces per minute as Kalinda." Yep, so if Kalinda is 4, then half would be 2. That makes sense. So, their rates are 4 and 2 pieces per minute respectively.

Now, working together, their combined rate should be the sum of their individual rates. So, 4 + 2 = 6 pieces per minute. Together, they can place 6 pieces each minute.

Then, the total number of pieces is 360. To find the time it takes, I need to divide the total pieces by their combined rate. So, time = total pieces / rate. That would be 360 / 6. Let me calculate that. 360 divided by 6 is 60. So, 60 minutes. But the question asks for the time in hours. So, 60 minutes is equal to 1 hour.

Wait, that seems straightforward. Let me verify again. Kalinda at 4 per minute, mom at 2 per minute, combined 6 per minute. 360 divided by 6 is 60 minutes, which is 1 hour. That seems correct.

But let me double-check the arithmetic. 6 pieces per minute times 60 minutes would be 360 pieces. Yes, that's exactly the number of pieces in the puzzle. So, 60 minutes, which is 1 hour. So, the answer is 1 hour.

Is there another way to approach this problem? Maybe convert their rates to pieces per hour first? Let's see. Kalinda does 4 pieces per minute. There are 60 minutes in an hour, so 4 * 60 = 240 pieces per hour. Her mom does 2 pieces per minute, so 2 * 60 = 120 pieces per hour. Together, their combined rate is 240 + 120 = 360 pieces per hour. Therefore, 360 pieces divided by 360 pieces per hour is 1 hour. Yep, same result. That confirms it.

So, whether I calculate in minutes and then convert to hours, or convert their rates to hours first, I end up with 1 hour. So, that must be the correct answer.

I don't see any other factors here. The problem states they are working on the puzzle together, so I assume they work simultaneously without any breaks or decreasing efficiency. The problem doesn't mention any time lost due to searching for pieces or anything like that, so it's a straightforward combined work rate problem.

Just to recap: Individual rates added together give the combined rate. Total work divided by combined rate gives time. Convert time to required units (hours). All steps check out. The answer is 1 hour.

So the answer is $\\boxed{1}$.""")

demo.append("Okay, let's see. The problem is about Natalia selling clips in April and May. So, in April, she sold clips to 48 of her friends. Then in May, she sold half as many clips. The question is asking how many clips she sold altogether in April and May. Hmm, let's break this down step by step.\n\nFirst, let's make sure I understand the problem correctly. In April, Natalia sold clips to 48 friends. Now, does that mean each friend bought one clip? Or did each friend buy multiple clips? The problem says she sold clips to 48 friends. Hmm, the wording here is a bit ambiguous. \"Sold clips to 48 friends\" could be interpreted in two ways: either she sold one clip per friend, totaling 48 clips, or she sold an unspecified number of clips to each of 48 friends, meaning the total number isn't specified. But the problem doesn't give us any more details about the number of clips per friend in April. Wait, but then it says she sold half as many clips in May. So maybe the 48 refers to the number of clips, not the number of friends? Wait, let me check the original problem again.\n\nOriginal problem: \"Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\"\n\nHmm, so it says she sold clips to 48 friends in April. So, 48 friends. Then in May, she sold half as many clips. So, \"half as many clips\" as in April. Wait, but in April, the problem mentions she sold to 48 friends, not necessarily 48 clips. So does \"half as many clips\" refer to half the number of clips or half the number of friends?\n\nThis is a bit confusing. Maybe we need to make an assumption here. Since the problem mentions \"clips\" in both months, but in April it specifies the number of friends she sold to, whereas in May it just says \"half as many clips.\" So perhaps the key here is that in April, she sold clips to 48 friends, but we don't know how many clips each friend bought. Then in May, she sold half as many clips as she did in April. Wait, but if we don't know how many clips she sold in April, how can we determine the number for May?\n\nWait, perhaps there's a misinterpretation here. Maybe the problem is intended to mean that she sold clips to 48 friends in April, and each friend bought one clip. So, 48 clips in April. Then in May, she sold half as many clips, so 24 clips. Then total would be 48 + 24 = 72 clips. But is that the correct interpretation?\n\nAlternatively, maybe \"sold clips to 48 friends\" means that each transaction was with a friend, but each friend could have bought multiple clips. But since the problem doesn't specify the number of clips per friend, maybe we have to assume that each friend bought one clip. That would be a standard assumption in such problems unless stated otherwise. So, in April, 48 clips, one per friend. Then in May, half as many clips, so 24. Total 72. That seems straightforward.\n\nBut let me check again. If it's 48 friends in April, and in May, half as many clips. If \"half as many clips\" refers to half the number of clips sold in April, then if we don't know the number of clips in April, we can't compute May's number. Therefore, the problem must mean that the number of clips sold in April is 48, which is sold to 48 friends (each buying one clip). Then May is half of that, so 24. Then total is 72. That makes sense.\n\nAlternatively, if \"sold clips to 48 friends\" in April, but each friend bought multiple clips, but since the problem mentions \"how many clips did Natalia sell altogether,\" and the answer would require knowing the number of clips. But since the problem doesn't specify the number of clips per friend, we have to assume that each friend bought one clip. Otherwise, the problem is unsolvable as we don't have enough information.\n\nTherefore, the problem is expecting us to interpret \"sold clips to 48 friends\" as 48 clips, one per friend, in April. Then in May, she sold half as many clips, so 24. Total is 48 + 24 = 72. So the answer is 72.\n\nWait, but let me verify again. The problem says: \"Natalia sold clips to 48 of her friends in April.\" So, she sold clips to 48 friends. So, the number of friends is 48, but the number of clips is not specified. Then in May, she sold half as many clips. If \"half as many clips\" as in April, then we need the number of clips sold in April. But we don't know that because the problem only mentions the number of friends. So unless each friend bought one clip, we can't get the number of clips.\n\nBut if we don't assume that, the problem is unsolvable. Therefore, the problem must be intending that \"sold clips to 48 friends\" means 48 clips, with each friend buying one. That's a common phrasing in math problems, where \"sold to x friends\" implies x units if the item is something typically bought one per person, like maybe a hair clip. So, perhaps each friend bought one clip.\n\nAlternatively, maybe in April, she sold 48 clips in total, regardless of the number of friends. Wait, the problem says \"sold clips to 48 of her friends in April.\" So the number of friends is 48. So it's possible that she sold multiple clips to each friend, but again, without additional information, we can't compute the total clips. Therefore, in a math problem context, unless specified otherwise, \"sold to x friends\" would usually imply that each friend bought one item, especially if the item is something like a clip which you might buy one of. So 48 friends, 48 clips. Then May is half of April's clips, so 24. Total 72. So I think that's the answer they are expecting.\n\nAlternatively, maybe \"half as many clips in May\" refers to half the number of transactions or half the number of friends. But since it says \"half as many clips,\" not \"half as many friends,\" so the number of clips is halved. So if in April, she sold 48 clips (to 48 friends, one each), then May is 24 clips. So total 72.\n\nSo the answer is \\boxed{72}.")
demo.append("Okay, let's see. Weng earns $12 per hour for babysitting. Yesterday, she babysat for 50 minutes. The question is asking how much she earned. Hmm, so I need to convert 50 minutes into hours because her rate is given per hour.\n\nFirst, I remember that 1 hour is 60 minutes. So, 50 minutes is less than an hour. To convert minutes to hours, I can divide the number of minutes by 60. Let me write that down: 50 minutes divided by 60 minutes per hour. So, 50/60. Let me calculate that. 50 divided by 60. Hmm, 50 divided by 60 is the same as 5/6, right? Because both numerator and denominator can be divided by 10. 50 divided by 10 is 5, 60 divided by 10 is 6. So, 5/6 of an hour.\n\nNow, her hourly rate is $12. So, she earns $12 for each hour. Since she worked 5/6 of an hour, I need to multiply her hourly rate by the fraction of the hour she worked. So, $12 times 5/6. Let me do that calculation. 12 multiplied by 5 is 60, and then divided by 6. 60 divided by 6 is 10. So, that would be $10.\n\nWait, let me check that again. 12 times 5 is 60, divided by 6 gives 10. Yeah, that seems right. Alternatively, maybe I can break it down differently. 12 divided by 6 is 2, and then multiplied by 5 is 10. Either way, the result is the same. So, she earned $10 for 50 minutes of babysitting.\n\nBut let me think if there's another way to approach this. Sometimes, with time conversions, it's easy to make a mistake. Let's confirm. If she makes $12 in 60 minutes, how much does she make per minute? So, 12 divided by 60 is 0.2 dollars per minute. Then, for 50 minutes, it would be 0.2 times 50. 0.2 times 50 is 10. Yeah, that also gives $10. So, two different methods give the same answer. That seems solid.\n\nAlternatively, maybe using decimal hours instead of fractions. 50 minutes is what in decimal hours? Since 50 minutes is 50/60 hours, which is approximately 0.8333 hours. Then, multiplying that by $12. Let's compute 12 times 0.8333. 12 times 0.8 is 9.6, and 12 times 0.0333 is approximately 0.3996. Adding those together, 9.6 + 0.3996 ≈ 10. So, approximately $10. That checks out too.\n\nSo, all methods lead me to believe that the correct amount she earned is $10. I don't see any errors in my calculations here. The key was converting the time she worked into hours to match the unit of her pay rate. Whether using fractions, decimals, or calculating per minute rate, all confirm the same result. Therefore, I'm confident the answer is $10. \n\nSo the answer is $\\boxed{10}$")

new_demo = []
for item in demo:
    words = item.split()
    new_demo.append(' '.join(['xxx' for i in range(len(words))]))
demo = new_demo

prompt = (
    f"Consider the following three reasoning process examples:\n" 
    + "Example 1 \n" + demo[0] + '\n'
    + "Example 2 \n" + demo[1] + '\n'
    + "Example 3 \n" + demo[2] + "\n\n"
    "Please refer to the reasoning approaches demonstrated above, solve the following question by yourself:\n"
    "{question}\n\n"
    "Finally, provide your answer in the following format: 'So the answer is $\\boxed{}$'."
)

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=prompt),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096,stopping_criteria=stop_words)
)

gsm8k_eval_cfg = dict(
    evaluator=dict(type=Gsm8kEvaluator),
    pred_postprocessor=dict(type=gsm8k_postprocess),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
)

gsm8k_datasets = [
    dict(
        abbr='gsm8k_slow_think_3shot_xxx',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
