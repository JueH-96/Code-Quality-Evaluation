def solve():
    n = int(input())
    p = list(map(int, input().split()))

    ans = 0
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] > p[j]:
                ans += (i + 1)

    print(ans)

solve()