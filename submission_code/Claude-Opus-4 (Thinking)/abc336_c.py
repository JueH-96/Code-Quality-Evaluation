# YOUR CODE HERE
n = int(input())

# Convert n-1 to base 5
n -= 1
digits = []
if n == 0:
    digits = [0]
else:
    while n > 0:
        digits.append(n % 5)
        n //= 5

# Map to even digits
mapping = [0, 2, 4, 6, 8]
result = 0
for i in range(len(digits) - 1, -1, -1):
    result = result * 10 + mapping[digits[i]]

print(result)