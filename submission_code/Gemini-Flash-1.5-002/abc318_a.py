# YOUR CODE HERE
n, m, p = map(int, input().split())

count = 0
if m > n:
  print(0)
else:
  count = (n - m) // p + 1
  print(count)