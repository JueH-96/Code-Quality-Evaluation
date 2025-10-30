def solve():
    n, m, c, k = map(int, input().split())
    a = list(map(int, input().split()))

    total_min_sum = 0
    for current_k in range(k):
        min_val = float('inf')
        for i in range(n):
            val = (c * current_k + a[i]) % m
            min_val = min(min_val, val)
        total_min_sum += min_val
    print(total_min_sum)

solve()