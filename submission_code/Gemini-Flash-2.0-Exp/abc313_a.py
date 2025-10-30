N = int(input())
P = list(map(int, input().split()))

max_other = 0
for i in range(1, N):
  max_other = max(max_other, P[i])

if P[0] > max_other:
  print(0)
else:
  print(max_other - P[0] + 1)