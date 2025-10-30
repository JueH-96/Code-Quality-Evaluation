N, X = map(int, input().split())
A = list(map(int, input().split()))

A.sort()
score_sum = sum(A)
grade_sum = X

low_goal = grade_sum - (score_sum - A[-1])
up_goal = grade_sum - (score_sum - A[0])

low_val = max(0, low_goal)
up_val = up_goal

answer = 101

if low_val <= up_val:
  if low_val <= 100:
    answer = low_val
elif low_val > up_val:
  answer = -1

print(answer)