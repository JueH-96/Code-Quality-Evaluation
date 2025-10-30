import math

def calculate_rating(performances):
    k = len(performances)
    if k == 0:
        return float('-inf')
    
    # Calculate weighted sum
    weighted_sum = 0
    for i in range(k):
        weighted_sum += (0.9 ** (k - 1 - i)) * performances[i]
    
    # Calculate denominator (geometric series)
    denominator = (1 - 0.9 ** k) / 0.1
    
    # Calculate rating
    rating = weighted_sum / denominator - 1200 / math.sqrt(k)
    return rating

# Read input
N = int(input())
P = list(map(int, input().split()))

# Try all possible contiguous subsequences
max_rating = float('-inf')

for start in range(N):
    for end in range(start, N):
        # Get subsequence from start to end (inclusive)
        subsequence = P[start:end+1]
        rating = calculate_rating(subsequence)
        max_rating = max(max_rating, rating)

print(f"{max_rating:.15f}")