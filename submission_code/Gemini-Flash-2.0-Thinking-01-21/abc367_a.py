a, b, c = map(int, input().split())

if b < c:
  if (a < b) or (a >= c):
    print("Yes")
  else:
    print("No")
else: # b > c
  if (c <= a) and (a < b):
    print("Yes")
  else:
    print("No")