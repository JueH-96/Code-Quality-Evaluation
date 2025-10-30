A, B = map(int, input().split())

if B == A + 1 and A % 3 != 0:
  print("Yes")
else:
  print("No")