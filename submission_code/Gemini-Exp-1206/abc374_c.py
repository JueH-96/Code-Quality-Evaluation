def solve():
    n = int(input())
    k = list(map(int, input().split()))

    ans = float('inf')
    for i in range(1 << n):
        group_a = 0
        group_b = 0
        for j in range(n):
            if (i >> j) & 1:
                group_a += k[j]
            else:
                group_b += k[j]
        ans = min(ans, max(group_a, group_b))

    print(ans)

solve()