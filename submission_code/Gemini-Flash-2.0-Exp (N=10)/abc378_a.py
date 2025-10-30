def solve():
    a = list(map(int, input().split()))
    counts = {}
    for x in a:
        counts[x] = counts.get(x, 0) + 1
    
    ans = 0
    for count in counts.values():
        ans += count // 2
    print(ans)

solve()