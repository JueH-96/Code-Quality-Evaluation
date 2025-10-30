# YOUR CODE HERE
def solve():
  n, k = map(int, input().split())
  a = list(map(int, input().split()))
  
  prod = 1
  for x in a:
    prod *= x
  
  if prod % k == 0:
    print(0)
    return

  ans = float('inf')
  
  if k == 2:
    min_ops = float('inf')
    for i in range(n):
      ops = (2 - (a[i] % 2))
      min_ops = min(min_ops, ops)
    ans = min_ops

  elif k == 3:
    min_ops = float('inf')
    for i in range(n):
      ops = (3 - (a[i] % 3))
      min_ops = min(min_ops, ops)
    ans = min_ops

  elif k == 4:
    min_ops = float('inf')
    
    count_2 = 0
    count_other = 0
    for x in a:
      if x % 4 == 0:
        continue
      elif x % 2 == 0:
        count_2 +=1
      else:
        count_other +=1
    
    if count_2 >= 1 and count_other >=1:
      ans = min(ans,1)
    elif count_2 >= 2:
      ans = min(ans,0)
    elif count_other >= 1:
      ans = min(ans, 3)
    else:
      min_ops = float('inf')
      for i in range(n):
        ops = (4 - (a[i] % 4))
        min_ops = min(min_ops, ops)
      ans = min_ops

  elif k == 5:
    min_ops = float('inf')
    for i in range(n):
      ops = (5 - (a[i] % 5))
      min_ops = min(min_ops, ops)
    ans = min_ops

  print(ans)


t = int(input())
for _ in range(t):
  solve()