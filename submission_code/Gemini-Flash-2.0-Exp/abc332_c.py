def solve():
    n, m = map(int, input().split())
    s = input()
    
    logo_needed = 0
    plain_available = m
    logo_available = 0
    
    for i in range(n):
        if s[i] == '1':
            if plain_available > 0:
                plain_available -= 1
            elif logo_available > 0:
                logo_available -= 1
            else:
                logo_needed += 1
        elif s[i] == '2':
            if logo_available > 0:
                logo_available -= 1
            else:
                logo_needed += 1
        elif s[i] == '0':
            plain_available = m
            logo_available = logo_needed
            logo_needed = 0
            
    print(logo_needed)

solve()