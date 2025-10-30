def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    ans = float('inf')
    for i in range(1 << n):
        subset = []
        subset_indices = []
        for j in range(n):
            if (i >> j) & 1:
                subset.append(a[j])
                subset_indices.append(j)

        if len(subset) == k:
            subset_b_sum = sum(b[j] for j in subset_indices)
            ans = min(ans, max(subset) * subset_b_sum)

    print(ans)


t = int(input())
for _ in range(t):
    solve()