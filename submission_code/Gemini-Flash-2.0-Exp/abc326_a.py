x, y = map(int, input().split())

diff = y - x

if abs(diff) <= 2 and diff > 0:
  print("Yes")
elif abs(diff) <= 3 and diff < 0:
  print("Yes")
else:
  print("No")