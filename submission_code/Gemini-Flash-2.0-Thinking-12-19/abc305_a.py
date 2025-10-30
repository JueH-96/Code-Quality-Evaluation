n = int(input())
remainder = n % 5
if remainder <= 2:
    result = n - remainder
else:
    result = n - remainder + 5
print(result)