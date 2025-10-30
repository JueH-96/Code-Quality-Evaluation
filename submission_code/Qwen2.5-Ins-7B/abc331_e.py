# NUMBER OF MAIN DISHES, SIDE DISHES AND UNOFFERED PAIRS
N, M, L = map(int, input().split())

# PRICES OF MAIN DISHES
a = [0] + list(map(int, input().split()))

# PRICES OF SIDE DISHES
b = [0] + list(map(int, input().split()))

# UNOFFERED PAIRS
unoffered = set()
for _ in range(L):
    c, d = map(int, input().split())
    unoffered.add((c, d))

# FIND THE MAX PRICE OF AN OFFERED SET MEAL
max_price = 0
for i in range(1, N+1):
    for j in range(1, M+1):
        if (i, j) not in unoffered:
            max_price = max(max_price, a[i] + b[j])

print(max_price)