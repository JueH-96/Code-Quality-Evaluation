# YOUR CODE HERE
N, X = map(int, input().split())
A = list(map(int, input().split()))

# Try each possible score from 0 to 100
for score in range(101):
    # Create a list with all scores including the new one
    all_scores = A + [score]
    
    # Sort the scores
    all_scores.sort()
    
    # Calculate final grade (sum of all except lowest and highest)
    final_grade = sum(all_scores[1:-1])
    
    # If this gives us the required grade, this is our answer
    if final_grade >= X:
        print(score)
        break
else:
    # If no score from 0 to 100 works, print -1
    print(-1)