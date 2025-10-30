N = int(input())

if N <= 999:
  print(N)
elif 1000 <= N <= 9999:
  print(N - (N % 10))
elif 10000 <= N <= 99999:
  print(N - (N % 100))
elif 100000 <= N <= 999999:
  print(N - (N % 1000))
elif 1000000 <= N <= 9999999:
  print(N - (N % 10000))
elif 10000000 <= N <= 99999999:
  print(N - (N % 100000))
elif 100000000 <= N <= 999999999:
  print(N - (N % 1000000))