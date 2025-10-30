def solve():
  n, x, y, z = map(int, input().split())

  if x < y:
    # Inbound train
    if x < z <= y:
      print("Yes")
    else:
      print("No")
  else:
    # Outbound train
    if y <= z < x:
      print("Yes")
    else:
      print("No")

solve()