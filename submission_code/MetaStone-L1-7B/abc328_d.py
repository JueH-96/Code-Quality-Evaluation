def solve():
    S = input().strip()
    s = list(S)
    i = 0
    while i < len(s) - 2:
        if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
            del s[i:i+3]
            i = max(i - 2, 0)
        else:
            i += 1
    print(''.join(s))

solve()