# YOUR CODE HERE
n = int(input())

if n < 10**3:
  print(n)
elif n < 10**4:
  print(n // 10 * 10)
elif n < 10**5:
  print(n // 100 * 100)
elif n < 10**6:
  print(n // 1000 * 1000)
elif n < 10**7:
  print(n // 10000 * 10000)
elif n < 10**8:
  print(n // 100000 * 100000)
else:
  print(n // 1000000 * 1000000)