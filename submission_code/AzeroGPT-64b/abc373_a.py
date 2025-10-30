N = 12
s = [input() for _ in range(N)]

ans = 0
for i in range(1,N+1):
  if len(s[i-1]) == i :
    ans += 1
print(ans)