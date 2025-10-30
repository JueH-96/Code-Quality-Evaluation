n = int(input())
s = input()
ans = 0
for i in range(n):
    for j in range(i, n):
        ans += int(s[i:j+1])
print(ans)