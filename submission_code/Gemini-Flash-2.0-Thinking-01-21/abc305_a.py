n = int(input())
remainder = n % 5
quotient = n // 5
if remainder <= 2:
    result = 5 * quotient
else:
    result = 5 * (quotient + 1)
print(result)