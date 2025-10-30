import sys

def solve(n, a):
    max_diff = 0
    for k in range(1, n + 1):
        if n % k == 0:
            truck_weights = [sum(a[i * k:(i + 1) * k]) for i in range(n // k)]
            max_diff = max(max_diff, max(truck_weights) - min(truck_weights))
    return max_diff

t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    print(solve(n, a))