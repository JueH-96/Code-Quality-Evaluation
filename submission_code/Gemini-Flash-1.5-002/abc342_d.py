import math

def is_square(n):
    if n < 0:
        return False
    sqrt_n = int(math.sqrt(n))
    return sqrt_n * sqrt_n == n

n = int(input())
a = list(map(int, input().split()))

count = 0
for i in range(n):
    for j in range(i + 1, n):
        if is_square(a[i] * a[j]):
            count += 1

print(count)