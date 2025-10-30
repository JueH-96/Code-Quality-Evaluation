a, b, c = map(int, input().split())

if b < c:
  if b < a < c:
    print("Yes")
  else:
    print("No")
else:
  if a < c or a > b:
    print("Yes")
  else:
    print("No")