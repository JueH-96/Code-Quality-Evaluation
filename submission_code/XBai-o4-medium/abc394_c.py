s = list(input())
i = 0
while i < len(s) - 1:
    if s[i] == 'W' and s[i+1] == 'A':
        s[i] = 'A'
        s[i+1] = 'C'
        i = max(0, i - 1)
    else:
        i += 1
print(''.join(s))