N, T = map(int, input().split())
scores = [0] * N
score_count = {0: N}  # Initially all N players have score 0

for _ in range(T):
    A, B = map(int, input().split())
    A -= 1  # Convert to 0-indexed
    old_score = scores[A]
    new_score = old_score + B
    
    # Update score_count dictionary
    score_count[old_score] -= 1
    if score_count[old_score] == 0:
        del score_count[old_score]
    
    if new_score in score_count:
        score_count[new_score] += 1
    else:
        score_count[new_score] = 1
    
    scores[A] = new_score
    print(len(score_count))