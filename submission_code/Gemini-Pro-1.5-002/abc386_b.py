s = input()
n = len(s)
ans = 0
i = 0
while i < n:
    if i + 1 < n and s[i] == '0' and s[i + 1] == '0':
        ans += 1
        i += 2
    elif i + 1 < n and s[i] != '0' and s[i+1] == '0' and i + 2 < n and s[i+2] == '0':
        ans += 2
        i += 3
    
    elif i + 1 < n and s[i] == '1' and s[i+1] == '0' and i + 2 < n and s[i+2] == '0':
        ans += 2
        i += 3
    else:
        ans += 1
        i += 1

print(ans)