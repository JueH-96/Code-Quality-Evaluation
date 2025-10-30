def cross_product(x1, y1, x2, y2, x3, y3):
  return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

def intersect(p1, q1, p2, q2):
  x1, y1 = p1
  x2, y2 = q1
  x3, y3 = p2
  x4, y4 = q2
  
  d1 = cross_product(x3, y3, x4, y4, x1, y1)
  d2 = cross_product(x3, y3, x4, y4, x2, y2)
  d3 = cross_product(x1, y1, x2, y2, x3, y3)
  d4 = cross_product(x1, y1, x2, y2, x4, y4)
  
  if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
    return True
  return False

def solve():
  n = int(input())
  p_points = []
  for _ in range(n):
    x, y = map(int, input().split())
    p_points.append((x, y))
  
  q_points = []
  for _ in range(n):
    x, y = map(int, input().split())
    q_points.append((x, y))
  
  import itertools
  
  for perm in itertools.permutations(range(n)):
    valid = True
    for i in range(n):
      for j in range(i + 1, n):
        if intersect(p_points[i], q_points[perm[i]], p_points[j], q_points[perm[j]]):
          valid = False
          break
      if not valid:
        break
    
    if valid:
      result = [x + 1 for x in perm]
      print(*result)
      return
  
  print("-1")

solve()