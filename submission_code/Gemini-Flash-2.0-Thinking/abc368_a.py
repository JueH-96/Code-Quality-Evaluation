def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    bottom_k = a[n-k:]
    remaining = a[:n-k]

    new_stack = bottom_k + remaining
    print(*new_stack)

solve()