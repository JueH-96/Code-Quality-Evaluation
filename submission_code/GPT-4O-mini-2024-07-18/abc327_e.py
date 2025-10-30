def max_rating(N, performances):
    max_rating_value = float('-inf')
    
    for k in range(1, N + 1):
        for start in range(N - k + 1):
            selected = performances[start:start + k]
            weighted_sum = sum((0.9 ** (k - i - 1)) * selected[i] for i in range(k))
            denominator = sum(0.9 ** (k - i - 1) for i in range(k))
            rating = (weighted_sum / denominator) - (1200 / (k ** 0.5))
            max_rating_value = max(max_rating_value, rating)
    
    return max_rating_value

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
performances = list(map(int, data[1:N + 1]))

result = max_rating(N, performances)
print(f"{result:.12f}")