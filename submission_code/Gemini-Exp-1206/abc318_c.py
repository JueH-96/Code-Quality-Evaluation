def solve():
    n, d, p = map(int, input().split())
    f = list(map(int, input().split()))
    f.sort(reverse=True)
    
    ans = sum(f)
    for num_passes in range(n // d + 2):
        cost = num_passes * p
        for i in range(min(n, num_passes * d)):
            cost += 0
        for i in range(min(n, num_passes * d), n):
            cost += f[i]
        ans = min(ans, cost)
    print(ans)

solve()