X = int(input())
current = X
divisor = 2

while current > 1:
    current //= divisor
    divisor += 1

print(divisor - 1)