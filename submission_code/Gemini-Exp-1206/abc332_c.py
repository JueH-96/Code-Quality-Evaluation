def solve():
    n, m = map(int, input().split())
    s = input()
    
    plain = m
    logo = 0
    ans = 0
    
    for char in s:
        if char == '0':
            plain = m
            logo = ans
        elif char == '1':
            if plain > 0:
                plain -= 1
            elif logo > 0:
                logo -= 1
            else:
                ans += 1
                logo = ans -1
                
        elif char == '2':
            if logo > 0:
                logo -= 1
            else:
                ans += 1
                logo = ans - 1
    
    print(ans)

solve()