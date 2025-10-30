def solve():
    n, m, k, c = map(int, input().split())
    a = list(map(int, input().split()))

    total_sum = 0
    for current_k in range(k):
        min_val = m
        for i in range(n):
            val = (c * current_k + a[i]) % m
            min_val = min(min_val, val)
        total_sum += min_val
    print(total_sum)

solve()