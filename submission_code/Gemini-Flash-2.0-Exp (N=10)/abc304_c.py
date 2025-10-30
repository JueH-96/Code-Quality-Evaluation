import math

def distance(x1, y1, x2, y2):
  return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def solve():
  n, d = map(int, input().split())
  coords = []
  for _ in range(n):
    coords.append(list(map(int, input().split())))
  
  infected = [False] * n
  infected[0] = True
  
  changed = True
  while changed:
    changed = False
    for i in range(n):
      if infected[i]:
        for j in range(n):
          if not infected[j]:
            if distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1]) <= d:
              infected[j] = True
              changed = True
  
  for i in range(n):
    print("Yes" if infected[i] else "No")

solve()