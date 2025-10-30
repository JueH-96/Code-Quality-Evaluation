# YOUR CODE HERE
N = int(input())
P = list(map(int, input().split()))

max_rating = -float('inf')

for i in range(1, 1 << N):
    chosen_contests = []
    for j in range(N):
        if (i >> j) & 1:
            chosen_contests.append(P[j])
    
    k = len(chosen_contests)
    sum_weighted_performance = 0
    sum_weights = 0
    for j in range(k):
        weight = (0.9)**(k - 1 - j)
        sum_weighted_performance += weight * chosen_contests[j]
        sum_weights += weight
    
    rating = sum_weighted_performance / sum_weights - 1200 / (k**0.5)
    max_rating = max(max_rating, rating)

print(max_rating)