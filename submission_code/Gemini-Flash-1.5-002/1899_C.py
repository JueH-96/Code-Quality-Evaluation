# YOUR CODE HERE
def solve():
  n = int(input())
  a = list(map(int, input().split()))

  max_sum = -float('inf')
  for i in range(n):
    current_sum = 0
    current_arr = []
    for j in range(i, n):
      current_arr.append(a[j])
      valid = True
      if len(current_arr) > 1:
        for k in range(len(current_arr) - 1):
          if (current_arr[k] % 2 == 0 and current_arr[k+1] % 2 == 0) or (current_arr[k] % 2 != 0 and current_arr[k+1] % 2 != 0):
            valid = False
            break
      if valid:
        current_sum = sum(current_arr)
        max_sum = max(max_sum, current_sum)

  if max_sum == -float('inf'):
    max_sum = max(a)

  print(max_sum)


t = int(input())
for _ in range(t):
  solve()