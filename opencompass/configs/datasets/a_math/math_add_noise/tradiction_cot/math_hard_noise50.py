from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MATHDataset, MATHEvaluator, math_postprocess_v2
import random
math_reader_cfg = dict(input_columns=['problem'], output_column='solution')
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>","<|end_of_text|>","\nught","\nughtuser","\nQuestion","\nnoteq","\nosterone","nosteroneuser"]
round = [
{'role': 'HUMAN', 'prompt': "Problem: If $\\mathbf{a},$ $\\mathbf{b},$ and $\\mathbf{c}$ are unit vectors, then find the largest possible value of\n\\[\\|\\mathbf{a} - \\mathbf{b}\\|^2 + \\|\\mathbf{a} - \\mathbf{c}\\|^2 + \\|\\mathbf{b} - \\mathbf{c}\\|^2.\\]Note: A unit vector is a vector of magnitude 1.\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': 'We can write\n\\begin{align*}\n\\|\\mathbf{a} - \\mathbf{b}\\|^2 &= (\\mathbf{a} - \\mathbf{b}) \\cdot (\\mathbf{a} - \\mathbf{b}) \\\\\n&= \\mathbf{a} \\cdot \\mathbf{a} - 2 \\mathbf{a} \\cdot \\mathbf{b} + \\mathbf{b} \\cdot \\mathbf{b} \\\\\n&= \\|\\mathbf{a}\\|^2 - 2 \\mathbf{a} \\cdot \\mathbf{b} + \\|\\mathbf{b}\\|^2 \\\\\n&= 2 - 2 \\mathbf{a} \\cdot \\mathbf{b}.\n\\end{align*}Similarly, $\\|\\mathbf{a} - \\mathbf{c}\\|^2 = 2 - 2 \\mathbf{a} \\cdot \\mathbf{c}$ and $\\|\\mathbf{b} - \\mathbf{c}\\|^2 = 2 - 2 \\mathbf{b} \\cdot \\mathbf{c},$ so\n\\[\\|\\mathbf{a} - \\mathbf{b}\\|^2 + \\|\\mathbf{a} - \\mathbf{c}\\|^2 + \\|\\mathbf{b} - \\mathbf{c}\\|^2 = 6 - 2 (\\mathbf{a} \\cdot \\mathbf{b} + \\mathbf{a} \\cdot \\mathbf{c} + \\mathbf{b} \\cdot \\mathbf{c}).\\]Now,\n\\[\\|\\mathbf{a} + \\mathbf{b} + \\mathbf{c}\\|^2 \\ge 0.\\]We can expand this as\n\\[\\|\\mathbf{a}\\|^2 + \\|\\mathbf{b}\\|^2 + \\|\\mathbf{c}\\|^2 + 2 \\mathbf{a} \\cdot \\mathbf{b} + 2 \\mathbf{a} \\cdot \\mathbf{c} + 2 \\mathbf{b} \\cdot \\mathbf{c} \\ge 0.\\]Then $2 (\\mathbf{a} \\cdot \\mathbf{b} + \\mathbf{a} \\cdot \\mathbf{c} + \\mathbf{b} \\cdot \\mathbf{c}) \\ge -3,$ so\n\\[\\|\\mathbf{a} - \\mathbf{b}\\|^2 + \\|\\mathbf{a} - \\mathbf{c}\\|^2 + \\|\\mathbf{b} - \\mathbf{c}\\|^2 = 6 - 2 (\\mathbf{a} \\cdot \\mathbf{b} + \\mathbf{a} \\cdot \\mathbf{c} + \\mathbf{b} \\cdot \\mathbf{c}) \\le 9.\\]Equality occurs when $\\mathbf{a},$ $\\mathbf{b},$ and $\\mathbf{c}$ are equally spaced on a circle with radius 1 (where $\\|\\mathbf{a} - \\mathbf{b}\\| = \\|\\mathbf{a} - \\mathbf{c}\\| = \\|\\mathbf{b} - \\mathbf{c}\\| = \\sqrt{3}$), so the largest possible value is $\\boxed{9}.$\n'},
{'role': 'HUMAN', 'prompt': "Problem: What is the largest five-digit integer whose digits have a product equal to the product $(7)(6)(5)(4)(3)(2)(1)$?\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': "Since we want a large 5-digit integer, we want the digits to the left to be as large as possible. We prime factorize the product to get $7 \\cdot 5 \\cdot 3^2 \\cdot 2^4$. The largest single-digit number is $9$, which can be found by $3^2$. This leaves us $7 \\cdot 5 \\cdot 2^4$. We can get the next largest number, $8$, by using $2^3$. This leaves $7 \\cdot 5\\cdot 2$. We can't multiply any of those numbers together to get a single-digit number, so we have that the remaining three digits are $7,5$, and $2$. Ordering the digits from largest to smallest, we get $\\boxed{98752}$.\n"},
{'role': 'HUMAN', 'prompt': "Problem: The number 236! ends in how many zeros when expressed as an integer?\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': 'To find the number of terminal zeros, we must find the number of products $2\\times5$ in $236!$. Since there are more factors of 2 than factors of 5, we can get our answer by finding the largest power of 5 that divides $236!$.  Every multiple of 5 less than 236 gives a factor of 5, each multiple of 25 gives an additional factor of 5, and each multiple of 125 gives a third factor of 5.  Therefore, the number of factors of 5 in $236!$ is $\\left\\lfloor\\frac{236}{5}\\right\\rfloor+ \\left\\lfloor\\frac{236}{25}\\right\\rfloor+ \\left\\lfloor\\frac{236}{125}\\right\\rfloor = 47+9+1=57$. The highest power of 5 that divides $236!$ is $5^{57}$ so $236!$ ends in $\\boxed{57}$ zeros.\n'},
{'role': 'HUMAN', 'prompt': "Problem: Find the positive solution to\n\\[\\sqrt[3]{x + \\sqrt[3]{x + \\sqrt[3]{x + \\dotsb}}} = \\sqrt[3]{x \\sqrt[3]{x \\sqrt[3]{x \\dotsm}}}.\\]\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': 'Let\n\\[y = \\sqrt[3]{x \\sqrt[3]{x \\sqrt[3]{x \\dotsm}}}.\\]Then\n\\[y^3 = x \\sqrt[3]{x \\sqrt[3]{x \\dotsm}} = xy,\\]so $y^2 = x.$\n\nLet\n\\[z = \\sqrt[3]{x + \\sqrt[3]{x + \\sqrt[3]{x + \\dotsb}}}.\\]Then\n\\[z^3 = x + \\sqrt[3]{x + \\sqrt[3]{x + \\dotsb}} = x + z,\\]so $z^3 - z = x.$\n\nSince $z = y,$ $y^3 - y = x = y^2.$  Then\n\\[y^3 - y^2 - y = 0,\\]which factors as $y (y^2 - y - 1) = 0,$ so $y^2 - y - 1 = 0.$  By the quadratic formula,\n\\[y = \\frac{1 \\pm \\sqrt{5}}{2}.\\]Since $y$ is positive,\n\\[y = \\frac{1 + \\sqrt{5}}{2}.\\]Then\n\\[x = y^2 = \\boxed{\\frac{3 + \\sqrt{5}}{2}}.\\]\n'},
{'role': 'HUMAN', 'prompt': "Problem: The graph of the quadratic $y = ax^2 + bx + c$ has the following properties: (1) The maximum value of $y = ax^2 + bx + c$ is 5, which occurs at $x = 3$. (2) The graph passes through the point $(0,-13)$. If the graph passes through the point $(4,m)$, then what is the value of $m$?\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': 'Since the maximum value of $y = ax^2 + bx + c$ is 5, which occurs at $x = 3$, this tells us that the vertex of the parabola is $(3,5)$.  Hence, the quadratic is of the form $y = a(x - 3)^2 + 5$, where $a$ is a negative number.  (We know that $a$ is negative because $y$ has a maximum value.)\n\nWe are also told that the graph passes through the point $(0,-13)$.  Substituting these coordinates into the equation $y = a(x - 3)^2 + 5$, we get $-13 = 9a + 5$, so $a = (-5 - 13)/9 = -18/9 = -2$.  Therefore, the equation is $y =- 2(x - 3)^2+5$.\n\nWhen $x = 4$, we have $m = - 2 \\cdot 1^2 + 5 = \\boxed{3}$.\n'},
{'role': 'HUMAN', 'prompt': "Problem: How many integers $n$ satisfy the condition $100 < n < 200$ and the condition $n$ has the same remainder whether it is divided by $6$ or by $8$?\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': 'Since $n$ has the same remainder whether divided by 6 or by 8, we can write that $n = 6a + r = 8b + r$, where $0\\leq r \\leq 5$.  This implies that $3a = 4b$, and so $a$ is a multiple of 4 and we can write $a = 4k$ for some integer $k$.  Since $100<n<200$, we see that $95<6a<200$, or $\\frac{95}{24} < k <\\frac{200}{24}$.  Since $k$ is an integer, $4\\leq k \\leq 8$.  If $k = 4$, then we must have $r = 5$.  Otherwise, $r = 0,1,2,3,4,5$ are all allowable.  Thus we have a total of $\\boxed{25}$ possible values for $n$.\n'},
{'role': 'HUMAN', 'prompt': "Problem: Two congruent right circular cones each with base radius $3$ and height $8$ have the axes of symmetry that intersect at right angles at a point in the interior of the cones a distance $3$ from the base of each cone. A sphere with radius $r$ lies withing both cones. The maximum possible value of $r^2$ is $\\frac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m+n$.\n\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': 'Consider the cross section of the cones and sphere by a plane that contains the two axes of symmetry of the cones as shown below. The sphere with maximum radius will be tangent to the sides of each of the cones. The center of that sphere must be on the axis of symmetry of each of the cones and thus must be at the intersection of their axes of symmetry. Let $A$ be the point in the cross section where the bases of the cones meet, and let $C$ be the center of the sphere. Let the axis of symmetry of one of the cones extend from its vertex, $B$, to the center of its base, $D$. Let the sphere be tangent to $\\overline{AB}$ at $E$. The right triangles $\\triangle ABD$ and $\\triangle CBE$ are similar, implying that the radius of the sphere is\\[CE = AD \\cdot\\frac{BC}{AB} = AD \\cdot\\frac{BD-CD}{AB} =3\\cdot\\frac5{\\sqrt{8^2+3^2}} = \\frac{15}{\\sqrt{73}}=\\sqrt{\\frac{225}{73}}.\\]The requested sum is $225+73=\\boxed{298}$.\n'},
{'role': 'HUMAN', 'prompt': "Problem: A rectangle is called cool if the number of square units in its area is equal to twice the number of units in its perimeter. A cool rectangle also must have integer side lengths. What is the sum of all the different possible areas of cool rectangles?\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
{'role': 'BOT', 'prompt': 'Let the side lengths of the rectangle be $a$ and $b.$ It follows that $ab=4(a+b).$ Expanding and moving all the terms to the left hand side, $ab-4a-4b=0.$ Adding 16 to both sides allows us to factor:\n\\[a(b-4)-4(b-4)=(a-4)(b-4)=16. \\]From this point, the pairs $(a,b)$ that provide different areas are $(5,20),$ $(6,12),$ and $(8,8),$ and the sum of the possible areas is $\\boxed{236}.$\n'},
{'role': 'HUMAN', 'prompt': "Problem:\n{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.\nSolution:"},
]

for item in round:
    if item['role'] == 'BOT':
        # 将字符串按空格分割成单词列表
        words = item['prompt'].split()
        # 计算需要替换的单词数量（50%）
        num_to_mask = int(len(words) * 0.5)
        # 随机选择需要替换的索引
        indices_to_mask = random.sample(range(len(words)), num_to_mask)
        # 将选中的单词替换为 "xxx"
        for index in indices_to_mask:
            words[index] = "xxx"
        # 将修改后的单词列表重新组合成字符串
        item['prompt'] = ' '.join(words)

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=round)),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096, stopping_criteria=stop_words))

math_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2))

math_datasets = [
    dict(
        type=MATHDataset,
        abbr='math_hard_noise50',
        path='opencompass/math',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg)
]
