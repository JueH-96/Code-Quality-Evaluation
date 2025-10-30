import math

def solve():
    n = int(input())
    p = list(map(int, input().split()))

    max_rating = float('-inf')

    for i in range(1, 1 << n):
        chosen_contests = []
        for j in range(n):
            if (i >> j) & 1:
                chosen_contests.append(p[j])
        
        k = len(chosen_contests)
        
        numerator = 0
        denominator = 0
        for idx, performance in enumerate(chosen_contests):
            weight = (0.9)**(k - 1 - idx)
            numerator += weight * performance
            denominator += weight
        
        if denominator == 0:
            continue
        
        rating = numerator / denominator - 1200 / math.sqrt(k)
        max_rating = max(max_rating, rating)
    
    print(max_rating)

solve()