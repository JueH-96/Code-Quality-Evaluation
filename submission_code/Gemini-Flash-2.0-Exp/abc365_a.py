Y = int(input())

if Y % 4 == 0:
  if Y % 100 == 0:
    if Y % 400 == 0:
      print(366)
    else:
      print(365)
  else:
    print(366)
else:
  print(365)