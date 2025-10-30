s = list(input().strip())
i = 0
while i <= len(s) - 3:
    if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
        del s[i:i+3]
        i = max(0, i-2)
    else:
        i += 1
print(''.join(s))