n = int(input())
a = list(map(int, input().split()))
l = [0]*n
r = [0]*n

for i in range(2*n):
  x = a[i]-1
  if l[x] == 0:
    l[x] = i
  else:
    r[x] = i

ans = 0

for i in range(n):
  if abs(l[i]-r[i]) == 2:
    ans += 1

print(ans)