def calculate_rating(performances):
    k = len(performances)
    if k == 0:
        return 0
    numerator = 0
    denominator = 0
    for i in range(k):
        numerator += (0.9)**(k - 1 - i) * performances[i]
        denominator += (0.9)**(k - 1 - i)
    return numerator / denominator - 1200 / (k**0.5)

def solve():
    n = int(input())
    performances = list(map(int, input().split()))
    
    max_rating = float('-inf')
    
    for i in range(1, 1 << n):
        chosen_performances = []
        for j in range(n):
            if (i >> j) & 1:
                chosen_performances.append(performances[j])
        
        max_rating = max(max_rating, calculate_rating(chosen_performances))
    
    print(max_rating)

solve()