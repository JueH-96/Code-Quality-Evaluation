# YOUR CODE HERE
R = int(input())

if R >= 1 and R <= 99:
  ans = 100 - R
if R >= 100 and R <= 199:
  ans = 200 - R
if R >= 200 and R <= 299:
  ans = 300 - R

print(ans)