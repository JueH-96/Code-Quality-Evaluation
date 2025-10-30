def solve():
    n = int(input())
    s = [input() for _ in range(n)]

    for k in range(1, n + 1):
        t = s[k-1]
        ans = len(t)
        
        for i in range(k - 1):
            other = s[i]
            
            l = 0
            while l < len(t) and l < len(other) and t[l] == other[l]:
                l += 1
            
            cost = len(t) - l + len(other) - l
            ans = min(ans, cost)
        
        print(ans)

solve()