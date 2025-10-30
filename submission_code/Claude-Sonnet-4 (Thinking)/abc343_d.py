from collections import defaultdict

n, t = map(int, input().split())
scores = [0] * n
score_count = defaultdict(int)
score_count[0] = n  # Initially, all players have score 0

for _ in range(t):
    a, b = map(int, input().split())
    old_score = scores[a - 1]
    new_score = old_score + b
    
    # Update the count
    score_count[old_score] -= 1
    if score_count[old_score] == 0:
        del score_count[old_score]
    
    score_count[new_score] += 1
    scores[a - 1] = new_score
    
    print(len(score_count))