# YOUR CODE HERE
N, X = map(int, input().split())
A = list(map(int, input().split()))

# Try each possible score for round N from 0 to 100
for score_N in range(101):  # 0 to 100 inclusive
    # Create the complete list of scores including round N
    all_scores = A + [score_N]
    
    # Sort the scores
    all_scores.sort()
    
    # Calculate final grade (sum excluding highest and lowest)
    final_grade = sum(all_scores[1:-1])  # Exclude first (lowest) and last (highest)
    
    # Check if this meets the requirement
    if final_grade >= X:
        print(score_N)
        exit()

# If we get here, no score from 0-100 works
print(-1)