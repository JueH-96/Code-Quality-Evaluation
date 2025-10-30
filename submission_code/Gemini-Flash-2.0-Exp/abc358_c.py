def solve():
    n, m = map(int, input().split())
    s = [input() for _ in range(n)]

    ans = n
    for i in range(1 << n):
        if i == 0:
            continue
        
        stands = []
        for j in range(n):
            if (i >> j) & 1:
                stands.append(j)
        
        flavors = [False] * m
        for stand_index in stands:
            for k in range(m):
                if s[stand_index][k] == 'o':
                    flavors[k] = True
        
        all_flavors = True
        for k in range(m):
            if not flavors[k]:
                all_flavors = False
                break
        
        if all_flavors:
            ans = min(ans, len(stands))
    
    print(ans)

solve()