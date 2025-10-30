N, R = map(int, input().split())

for _ in range(N):
  D, A = map(int, input().split())
  
  if D == 1:
    if 1600 <= R <= 2799:
      R += A
  else:
    if 1200 <= R <= 2399:
      R += A

print(R)