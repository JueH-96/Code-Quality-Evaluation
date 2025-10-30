def get_f(x):
  while x % 2 == 0:
    x //= 2
  return x

def solve():
  n = int(input())
  a = list(map(int, input().split()))
  total_sum = 0
  for i in range(n):
    for j in range(i, n):
      s = a[i] + a[j]
      f_val = get_f(s)
      total_sum += f_val
  print(total_sum)

if __name__ == '__main__':
  solve()