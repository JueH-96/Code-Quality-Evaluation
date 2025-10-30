def solve():
    N = int(input())
    A = list(map(int, input().split()))

    total = 0
    max_val = 0

    for i in range(N):
        max_val = max(max_val, A[i])
        total += max_val - A[i]

    print(total)

solve()