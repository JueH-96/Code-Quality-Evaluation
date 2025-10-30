# YOUR CODE HERE
def min_score_needed(N, X, scores):
    for potential_score in range(0, 101):
        # Add the potential score to our existing scores
        all_scores = scores + [potential_score]
        all_scores.sort()
        
        # Calculate final grade by excluding highest and lowest
        final_grade = sum(all_scores[1:-1])
        
        # Check if this score gives us the desired grade
        if final_grade >= X:
            return potential_score
    
    # If no score between 0-100 can achieve the desired grade
    return -1

# Read input
N, X = map(int, input().split())
scores = list(map(int, input().split()))

# Print the result
print(min_score_needed(N, X, scores))