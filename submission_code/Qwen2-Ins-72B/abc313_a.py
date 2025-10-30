N = int(input())
P = list(map(int, input().split()))

P1 = P[0]
Pmax = max(P[1:])
if P1 > Pmax:
  print(0)
else:
  print(Pmax - P1 + 1)