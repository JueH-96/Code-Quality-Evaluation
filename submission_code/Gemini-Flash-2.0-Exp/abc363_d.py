def generate_palindrome(n):
  s = str(n)
  return int(s + s[::-1])

def generate_odd_palindrome(n):
  s = str(n)
  return int(s + s[:-1][::-1])

def solve():
  n = int(input())
  if n == 1:
    print(0)
    return

  n -= 1
  
  k = 1
  count = 9
  while n > count:
    n -= count
    k += 1
    count = 9 * (10**(k-1))
  
  start = 10**(k-1)
  num = start + n - 1
  
  print(generate_odd_palindrome(num))

solve()