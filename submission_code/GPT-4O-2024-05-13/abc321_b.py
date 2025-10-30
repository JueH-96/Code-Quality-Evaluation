# YOUR CODE HERE
def min_score_for_final_grade(N, X, scores):
    scores.sort()
    current_sum = sum(scores[1:N-1])
    
    if current_sum >= X:
        return 0
    
    min_score_needed = X - current_sum
    if min_score_needed > 100:
        return -1
    return min_score_needed

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
X = int(data[1])
scores = list(map(int, data[2:]))

print(min_score_for_final_grade(N, X, scores))