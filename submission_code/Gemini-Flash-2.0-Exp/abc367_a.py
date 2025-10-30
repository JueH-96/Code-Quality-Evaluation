a, b, c = map(int, input().split())

if b < c:
  if a >= c or a < b:
    print("No")
  else:
    print("Yes")
else:
  if a >= c and a < b:
    print("No")
  else:
    print("Yes")