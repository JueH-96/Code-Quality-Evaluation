n = int(input())
n -= 1  # Convert to 0-indexed

digit_map = "02468"
result = ""

if n == 0:
    result = "0"
else:
    while n > 0:
        result = digit_map[n % 5] + result
        n //= 5

print(result)