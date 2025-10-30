import sys
import math

input = sys.stdin.read
data = input().split()

N = int(data[0])
P = list(map(int, data[1:]))

def calculate_rating(selected_performances):
    k = len(selected_performances)
    if k == 0:
        return float('-inf')
    
    weighted_sum = 0
    weight_total = 0
    for i, performance in enumerate(selected_performances):
        weight = 0.9 ** (k - i - 1)
        weighted_sum += weight * performance
        weight_total += weight
    
    average_weighted = weighted_sum / weight_total
    penalty = 1200 / math.sqrt(k)
    return average_weighted - penalty

max_rating = float('-inf')

# We will use a dynamic programming approach to find the optimal subset
# dp[i][j] will be the best rating achievable using the first i contests and exactly j of them
# However, storing all dp[i][j] is memory intensive, we will use a rolling array technique

# We will use a list to store the current best ratings for each possible number of contests used
current_ratings = [float('-inf')] * (N + 1)
current_ratings[0] = 0  # Base case: using 0 contests, the rating is 0 (not valid, but helps in setup)

for i in range(1, N + 1):
    new_ratings = [float('-inf')] * (N + 1)
    for j in range(1, i + 1):
        # Option 1: Include the i-th contest in the best of (j-1) from the first (i-1)
        if current_ratings[j-1] != float('-inf'):
            # Calculate the new rating if we include this contest
            selected_performances = []
            # Collect the last (j-1) performances that made the best score
            count = j-1
            index = i-2
            while count > 0 and index >= 0:
                selected_performances.append(P[index])
                index -= 1
                count -= 1
            selected_performances.append(P[i-1])
            selected_performances.reverse()
            
            new_rating = calculate_rating(selected_performances)
            new_ratings[j] = max(new_ratings[j], new_rating)
    
    # Update current_ratings to new_ratings
    current_ratings = new_ratings

# The answer is the maximum value in current_ratings
max_rating = max(current_ratings)

print(f"{max_rating:.12f}")