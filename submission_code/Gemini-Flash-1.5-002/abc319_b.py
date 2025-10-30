# YOUR CODE HERE
n = int(input())
divisors = []
for i in range(1, 10):
    if n % i == 0:
        divisors.append(i)
result = ""
for i in range(n + 1):
    min_divisor = float('inf')
    found = False
    for j in divisors:
        if i % (n // j) == 0:
            min_divisor = min(min_divisor, j)
            found = True
    if found:
        result += str(min_divisor)
    else:
        result += "-"
print(result)