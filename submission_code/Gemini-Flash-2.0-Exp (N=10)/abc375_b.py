import math

def calculate_distance(x1, y1, x2, y2):
  return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def solve():
  n = int(input())
  points = []
  for _ in range(n):
    x, y = map(int, input().split())
    points.append((x, y))
  
  total_cost = 0
  current_x, current_y = 0, 0
  
  for next_x, next_y in points:
    total_cost += calculate_distance(current_x, current_y, next_x, next_y)
    current_x, current_y = next_x, next_y
    
  total_cost += calculate_distance(current_x, current_y, 0, 0)
  
  print(total_cost)

solve()