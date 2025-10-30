import math

n = int(input())
p = list(map(int, input().split()))

max_rating = float('-inf')

# Try all possible contiguous subsequences
for start in range(n):
    for end in range(start, n):
        k = end - start + 1
        
        # Calculate weighted sum and weight sum
        weighted_sum = 0
        weight_sum = 0
        
        for i in range(k):
            weight = (0.9) ** (k - 1 - i)
            weighted_sum += weight * p[start + i]
            weight_sum += weight
        
        # Calculate rating
        rating = weighted_sum / weight_sum - 1200 / math.sqrt(k)
        max_rating = max(max_rating, rating)

print(max_rating)