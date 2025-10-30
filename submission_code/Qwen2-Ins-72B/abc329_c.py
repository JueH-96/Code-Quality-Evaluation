n = int(input())
s = input()

ans = 0
for i in range(n):
    if i == 0 or s[i] != s[i-1]:
        cnt = 1
    else:
        cnt += 1
    ans += cnt

print(ans)