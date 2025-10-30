N = int(input())
A = list(map(int, input().split()))

for i in range(N):
  week_sum = 0
  for j in range(7):
    week_sum += A[i * 7 + j]
  print(week_sum, end=" ")
print()