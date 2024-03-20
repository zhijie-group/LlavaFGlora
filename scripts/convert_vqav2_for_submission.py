import os
import argparse
import json
import pprint
from llava.eval.m4c_evaluator import EvalAIAnswerProcessor


'''主要用于处理生成的结果文件，将其转换为特定格式的答案文件'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="./playground/data/eval/vqav2")
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    # /home/bokai/LLaVA/playground/data/eval/vqav2/answers/llava_vqav2_mscoco_test-dev2015/llava-v1.5-13b/merge.jsonl
    src = os.path.join(args.dir, 'answers', args.split, args.ckpt, 'merge.jsonl')       
    test_split = os.path.join(args.dir, 'llava_vqav2_mscoco_test2015.jsonl')
    dst = os.path.join(args.dir, 'answers_upload', args.split, f'{args.ckpt}.json')
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    results = []
    error_line = 0
    for line_idx, line in enumerate(open(src)):
        try:
            results.append(json.loads(line))
            
            # pprint.pprint(json.loads(line))
#  {'answer_id': '7pkbASSfNyfTNMbFJFtpih',
#  'metadata': {},
#  'model_id': 'llava-v1.5-13b',
#  'prompt': 'Is the baby related to the goat?\n'
#            'Answer the question using a single word or phrase.',
#  'question_id': 550046000,
#  'text': 'Yes'}

        except:
            error_line += 1

    results = {x['question_id']: x['text'] for x in results}
    
    # 打开测试集文件(test_split)，将其中的每一行解析为 JSON 对象，并将问题ID添加到 split_ids 集合中
    test_split = [json.loads(line) for line in open(test_split)]
    split_ids = set([x['question_id'] for x in test_split])

    print(f'total results: {len(results)}, total split: {len(test_split)}, error_line: {error_line}')

    all_answers = []

    answer_processor = EvalAIAnswerProcessor()

    for x in test_split:
        if x['question_id'] not in results:
            all_answers.append({
                'question_id': x['question_id'],
                'answer': ''
            })
        else:
            all_answers.append({
                'question_id': x['question_id'],
                'answer': answer_processor(results[x['question_id']])
            })

    with open(dst, 'w') as f:
        json.dump(all_answers, open(dst, 'w'))
