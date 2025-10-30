s = input()
n = len(s)
ans = 0
t = ""
for char in s:
    if len(t) % 2 == 0:
        if char == 'i':
            t += char
        else:
            ans += 1
    else:
        if char == 'o':
            t += char
        else:
            ans += 1

if len(t) % 2 != 0:
    ans += 1

print(ans)