n = int(input())
s = input()

ans = 0
for i in range(1, n-1):
    if s[i-1] == '#' and s[i] == '.' and s[i+1] == '#':
        ans += 1

print(ans)