def dist_sq(x1, y1, x2, y2):
  return (x2 - x1)**2 + (y2 - y1)**2

xa, ya = map(int, input().split())
xb, yb = map(int, input().split())
xc, yc = map(int, input().split())

ab_sq = dist_sq(xa, ya, xb, yb)
bc_sq = dist_sq(xb, yb, xc, yc)
ca_sq = dist_sq(xc, yc, xa, ya)

if ab_sq + bc_sq == ca_sq or ab_sq + ca_sq == bc_sq or bc_sq + ca_sq == ab_sq:
  print("Yes")
else:
  print("No")