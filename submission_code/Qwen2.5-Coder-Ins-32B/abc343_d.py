# YOUR CODE HERE
import sys
from collections import defaultdict

input = sys.stdin.read().split()
N = int(input[0])
T = int(input[1])

scores = [0] * (N + 1)
score_count = defaultdict(int)
score_count[0] = N

index = 2
results = []

for _ in range(T):
    player = int(input[index])
    points = int(input[index + 1])
    index += 2
    
    old_score = scores[player]
    new_score = old_score + points
    scores[player] = new_score
    
    score_count[old_score] -= 1
    if score_count[old_score] == 0:
        del score_count[old_score]
    
    score_count[new_score] += 1
    
    results.append(len(score_count))

print("
".join(map(str, results)))