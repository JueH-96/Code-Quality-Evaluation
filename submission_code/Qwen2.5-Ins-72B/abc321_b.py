# YOUR CODE HERE
import sys

def min_score_for_final_grade(N, X, scores):
    scores.sort()
    min_score = 0
    max_score = 100
    current_sum = sum(scores[1:-1])
    
    for score in range(0, 101):
        new_scores = scores + [score]
        new_scores.sort()
        new_sum = sum(new_scores[1:-1])
        if new_sum >= X:
            return score
    return -1

input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
scores = list(map(int, data[2:]))

result = min_score_for_final_grade(N, X, scores)
print(result)