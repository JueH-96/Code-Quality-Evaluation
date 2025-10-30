def solve():
    n = int(input())
    slimes = {}
    for _ in range(n):
        s, c = map(int, input().split())
        slimes[s] = c

    sorted_keys = sorted(slimes.keys())
    
    for s in sorted_keys:
        c = slimes[s]
        if c >= 2:
            if 2 * s not in slimes:
                slimes[2 * s] = 0
            slimes[2 * s] += c // 2
            slimes[s] = c % 2

    ans = 0
    for s in slimes:
        ans += slimes[s]
    print(ans)

solve()