import sys

def solve_case():
    N, K = map(int, input().split())
    cakes = []
    for _ in range(N):
        X, Y, Z = map(int, input().split())
        cakes.append((X, Y, Z))

    cakes.sort(key=lambda x: (-max(x), x))
    total_price = 0
    for i in range(0, 2*K, 2):
        a, b = cakes[i], cakes[i+1]
        total_price += max(a[0] + b[0], a[1] + b[1], a[2] + b[2])

    return total_price

T = int(input())
for _ in range(T):
    print(solve_case())