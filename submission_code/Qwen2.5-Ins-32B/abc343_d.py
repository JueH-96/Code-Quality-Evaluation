# YOUR CODE HERE
import sys
input = sys.stdin.read
from collections import defaultdict

N, T, *AB = map(int, input().split())

scores = defaultdict(int)
unique_scores = set()
results = []

for a, b in zip(*[iter(AB)] * 2):
    prev_score = scores[a]
    scores[a] += b
    new_score = scores[a]
    
    if prev_score in unique_scores:
        unique_scores.remove(prev_score)
    unique_scores.add(new_score)
    
    results.append(len(unique_scores))

print('
'.join(map(str, results)))