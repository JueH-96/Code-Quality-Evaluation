n = int(input())

if n % 2 == 0:
  if n == 2:
    print("==")
  else:
    result = "-" * (n // 2 - 1) + "==" + "-" * (n // 2 - 1)
    print(result)
else:
  result = "-" * (n // 2) + "=" + "-" * (n // 2)
  print(result)