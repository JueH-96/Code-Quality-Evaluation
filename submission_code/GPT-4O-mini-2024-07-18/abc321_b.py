def minimum_score_needed(N, X, scores):
    scores.sort()
    
    # The maximum possible score we can achieve in round N
    max_possible_score = 100
    
    # Calculate the current sum of the scores
    current_sum = sum(scores)
    
    # We need to find the minimum score in round N (let's call it S_N)
    # The final grade is calculated as:
    # final_grade = (current_sum + S_N) - (max(S_N, max(scores)) + min(S_N, min(scores)))
    
    # We need to check if it's possible to achieve at least X
    # The minimum score we can earn in round N is 0
    # The maximum score we can earn in round N is 100
    
    # If we assume S_N = 0
    min_final_grade_with_0 = current_sum - max(scores) - min(scores)
    
    # If we assume S_N = 100
    min_final_grade_with_100 = current_sum + max_possible_score - max(max_possible_score, max(scores)) - min(max_possible_score, min(scores))
    
    # If the best possible score with S_N = 100 is still less than X, return -1
    if min_final_grade_with_100 < X:
        return -1
    
    # Now we need to find the minimum S_N such that the final grade is at least X
    for S_N in range(101):
        final_grade = current_sum + S_N - max(S_N, max(scores)) - min(S_N, min(scores))
        if final_grade >= X:
            return S_N
    
    return -1

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
scores = list(map(int, data[2:]))

# Get the result
result = minimum_score_needed(N, X, scores)

# Print the result
print(result)