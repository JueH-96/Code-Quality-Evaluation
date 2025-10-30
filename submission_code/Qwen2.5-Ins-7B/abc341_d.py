# YOUR CODE HERE
from math import gcd

def lcm(a, b):
    return a * b // gcd(a, b)

n, m, k = map(int, input().split())
l = lcm(n, m)
count = 0
i = 1

while True:
    if (i * n - 1) // m >= 0:
        count += (i * n - 1) // m - (i * m - 1) // n + 1
    else:
        count += (i * n - 1) // m + 1
    if count >= k:
        print(i * n - ((i * n - 1) // m - (i * m - 1) // n) * m)
        break
    i += 1