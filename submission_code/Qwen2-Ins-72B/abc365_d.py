n = int(input())
s = input()
s += 'X'
ans = 0
cnt = 1
for i in range(n):
    if s[i] != s[i+1]:
        if cnt >= 2:
            ans += (cnt-1)//3 * 2
            cnt = 1
        else:
            cnt += 1
    else:
        cnt += 1
if cnt >= 2:
    ans += (cnt-1)//3 * 2
print(ans)