# YOUR CODE HERE
N, T = map(int, input().split())

# Initialize scores for all players
scores = [0] * (N + 1)  # 1-indexed, so we need N+1 elements

# Dictionary to count occurrences of each score
score_counts = {0: N}  # Initially all N players have score 0

for _ in range(T):
    A, B = map(int, input().split())
    
    # Get the current score of player A
    old_score = scores[A]
    new_score = old_score + B
    
    # Update the score count dictionary
    # Decrease count for old score
    score_counts[old_score] -= 1
    if score_counts[old_score] == 0:
        del score_counts[old_score]
    
    # Increase count for new score
    if new_score in score_counts:
        score_counts[new_score] += 1
    else:
        score_counts[new_score] = 1
    
    # Update the player's score
    scores[A] = new_score
    
    # Output the number of distinct scores
    print(len(score_counts))