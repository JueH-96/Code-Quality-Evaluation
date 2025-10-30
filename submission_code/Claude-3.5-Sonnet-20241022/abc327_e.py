def calculate_rating(performances):
    n = len(performances)
    weights = [(0.9)**(n-i-1) for i in range(n)]
    numerator = sum(w * p for w, p in zip(weights, performances))
    denominator = sum(weights)
    return numerator/denominator - 1200/pow(n, 0.5)

N = int(input())
P = list(map(int, input().split()))

max_rating = float('-inf')
for mask in range(1, 1 << N):
    selected = []
    for i in range(N):
        if mask & (1 << i):
            selected.append(P[i])
    rating = calculate_rating(selected)
    max_rating = max(max_rating, rating)

print(max_rating)