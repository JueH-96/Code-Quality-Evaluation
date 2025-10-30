import math

def solve(n):
    if n == 1:
        return "1"
    if n < 10:
        return str(n) + "*" + str(n)
    for i in range(2, math.isqrt(n) + 1):
        if n % i == 0:
            other = n // i
            if str(i) == str(i)[::-1] and str(other) == str(other)[::-1]:
                return str(i) + "*" + str(other)
    return "-1"

n = int(input().strip())
print(solve(n))