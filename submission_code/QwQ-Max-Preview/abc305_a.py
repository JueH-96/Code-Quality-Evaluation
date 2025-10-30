n = int(input())
rem = n % 5
if rem <= 2:
    print(n - rem)
else:
    print(n + (5 - rem))