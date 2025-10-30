import math

x = int(input())

if x % 10 == 0:
    result = x // 10
else:
    result = (x // 10) + 1

print(result)