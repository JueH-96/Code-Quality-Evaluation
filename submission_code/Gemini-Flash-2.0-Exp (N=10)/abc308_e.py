def mex(a, b, c):
  s = set([a, b, c])
  for i in range(4):
    if i not in s:
      return i

def solve():
  n = int(input())
  a = list(map(int, input().split()))
  s = input()
  
  total_mex_sum = 0
  for i in range(n):
    for j in range(i + 1, n):
      for k in range(j + 1, n):
        if s[i] == 'M' and s[j] == 'E' and s[k] == 'X':
          total_mex_sum += mex(a[i], a[j], a[k])
  print(total_mex_sum)

solve()