import sys

t = int(sys.stdin.readline())
for _ in range(t):
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    max_product = 0
    for i in range(n):
        modified = a.copy()
        modified[i] += 1
        product = 1
        for num in modified:
            product *= num
        if product > max_product:
            max_product = product
    print(max_product)