def calculate_f(x):
  current_x = x
  while current_x % 2 == 0:
    current_x //= 2
  return current_x

def solve():
  n = int(input())
  a = list(map(int, input().split()))
  total_sum = 0
  for i in range(n):
    for j in range(i, n):
      sum_val = a[i] + a[j]
      f_val = calculate_f(sum_val)
      total_sum += f_val
  print(total_sum)

if __name__ == '__main__':
  solve()