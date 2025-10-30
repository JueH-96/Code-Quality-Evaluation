import math

def solve():
    n = int(input())
    performances = list(map(int, input().split()))
    
    dp = {} # dp[(i, j)] will store f(i, j)
    
    for i in range(1, n + 1):
        dp[(i, 1)] = performances[i-1]
        
    for j in range(2, n + 1):
        for i in range(j, n + 1):
            max_prev_val = -float('inf')
            for m in range(j - 1, i):
                max_prev_val = max(max_prev_val, dp.get((m + 1, j - 1), -float('inf')))
            if max_prev_val != -float('inf'):
                dp[(i, j)] = performances[i-1] + 0.9 * max_prev_val
            else:
                dp[(i, j)] = -float('inf')
                
    max_rating = -float('inf')
    
    for k in range(1, n + 1):
        max_numerator = -float('inf')
        for i in range(k, n + 1):
            max_numerator = max(max_numerator, dp.get((i, k), -float('inf')))
        if max_numerator != -float('inf'):
            denominator_sum = 10 * (1 - (0.9)**k)
            if denominator_sum == 0:
                average_performance = 0
            else:
                average_performance = max_numerator / denominator_sum
            rating = average_performance - (1200 / math.sqrt(k))
            max_rating = max(max_rating, rating)
            
    print(f"{max_rating:.17f}")

if __name__ == '__main__':
    solve()