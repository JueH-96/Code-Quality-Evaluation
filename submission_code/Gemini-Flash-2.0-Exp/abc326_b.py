def solve():
  n = int(input())
  
  for i in range(n, 1000):
    s = str(i)
    a = int(s[0])
    b = int(s[1])
    c = int(s[2])
    
    if a * b == c:
      print(i)
      return

solve()