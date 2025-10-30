def solve():
    n, m, d = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    max_sum = -1
    for i in range(n):
        for j in range(m):
            if abs(a[i] - b[j]) <= d:
                max_sum = max(max_sum, a[i] + b[j])

    print(max_sum)

solve()