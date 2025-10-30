def is_palindrome(n):
  s = str(n)
  return s == s[::-1]

def solve():
  n = int(input())
  ans = 0
  for i in range(1, 1000001):
    cube = i * i * i
    if cube > n:
      break
    if is_palindrome(cube):
      ans = cube
  print(ans)

solve()