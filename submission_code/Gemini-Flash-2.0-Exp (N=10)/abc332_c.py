def solve():
    n, m = map(int, input().split())
    s = input()
    
    plain_shirts = m
    logo_shirts = 0
    max_logo_shirts_needed = 0
    
    used_plain_shirts = 0
    used_logo_shirts = 0
    
    for day in s:
        if day == '0':
            used_plain_shirts = 0
            used_logo_shirts = 0
        elif day == '1':
            if used_plain_shirts < plain_shirts:
                used_plain_shirts += 1
            else:
                if used_logo_shirts < logo_shirts:
                    used_logo_shirts += 1
                else:
                    logo_shirts += 1
                    used_logo_shirts += 1
                    max_logo_shirts_needed = max(max_logo_shirts_needed, logo_shirts)
        elif day == '2':
            if used_logo_shirts < logo_shirts:
                used_logo_shirts += 1
            else:
                logo_shirts += 1
                used_logo_shirts += 1
                max_logo_shirts_needed = max(max_logo_shirts_needed, logo_shirts)
    
    print(max(0, max_logo_shirts_needed - (m if m > 0 else 0)))

solve()