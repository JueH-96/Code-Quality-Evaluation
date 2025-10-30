def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    min_val = float('inf')

    for i in range(1 << n):
        subset = []
        for j in range(n):
            if (i >> j) & 1:
                subset.append(j)

        if len(subset) == k:
            max_a = 0
            sum_b = 0
            for idx in subset:
                max_a = max(max_a, a[idx])
                sum_b += b[idx]
            
            min_val = min(min_val, max_a * sum_b)

    print(min_val)

t = int(input())
for _ in range(t):
    solve()