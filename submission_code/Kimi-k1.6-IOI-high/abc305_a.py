n = int(input())
remainder = n % 5
if remainder <= 2:
    print(n - remainder)
else:
    print(n + (5 - remainder))