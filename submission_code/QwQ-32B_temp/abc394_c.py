s = list(input().strip())
n = len(s)
i = 0
while i < n - 1:
    if s[i] == 'W' and s[i+1] == 'A':
        s[i] = 'A'
        s[i+1] = 'C'
        i -= 1
    else:
        i += 1
print(''.join(s))