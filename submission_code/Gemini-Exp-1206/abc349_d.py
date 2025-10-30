def solve():
    L, R = map(int, input().split())
    
    result = []
    
    l = L
    while l < R:
        r = l + 1
        power = 0
        while r * 2 <= R:
            if (l >> (power + 1)) == ((r * 2 - 1) >> (power + 1)):
                r *= 2
                power += 1
            else:
                break
        
        if r > R:
            r = R
        
        if (l >> power) == ((r - 1) >> power):
            result.append((l, r))
            l = r
        else:
            
            r = ( (l >> power) + 1) << power
            result.append((l,r))
            l = r
            
    print(len(result))
    for l, r in result:
        print(l, r)

solve()