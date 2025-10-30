m = int(input())
d = list(map(int,input().split()))

y = 0
for i in range(m):
  y += d[i]
  if y > (sum(d)+1)//2:
    a = i+1
    b = (sum(d)+1)//2 - sum(d[:i])
    break
print(a,b)