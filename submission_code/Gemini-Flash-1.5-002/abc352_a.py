# YOUR CODE HERE
n, x, y, z = map(int, input().split())

if x < y:
  if x < z and z < y:
    print("Yes")
  else:
    print("No")
else:
  if y < z and z < x:
    print("Yes")
  else:
    print("No")