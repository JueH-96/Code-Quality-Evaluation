N, A, B = map(int, input().split())
D = list(map(int, input().split()))

if D[0] <= A:
  flag = True
else:
  flag = False

for i in range(1, N):
  d = D[i] - D[i-1]
  if d > B:
    flag = False
    break
  elif d + D[i-1] > A + B:
    flag = False
    break

if flag:
  print("Yes")
else:
  print("No")