N,X = map(int,input().split())
A = list(map(int,input().split()))
A.sort()
s = sum(A[1:N-2])
if s + 100 < X:
  print(-1)
else:
  print(max(0,X-s))