n = int(input())
result = ""

for i in range(n+1):
    divisors = []
    for j in range(1, 10):
        if n % j == 0 and i % (n // j) == 0:
            divisors.append(j)
    if divisors:
        result += str(min(divisors))
    else:
        result += "-"

print(result)