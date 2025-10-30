# YOUR CODE HERE
import sys

def max_product(n, a):
    a.sort()
    for i in range(n):
        if a[i] != 0:
            a[i] += 1
            break
    product = 1
    for i in range(n):
        product *= a[i]
    return product

t = int(input().strip())
for _ in range(t):
    n = int(input().strip())
    a = list(map(int, input().strip().split()))
    print(max_product(n, a))