MOD = 10**9 + 7

def maximum_product(a, b, n):
    c = a ^ b
    u = 0
    for i in range(n):
        if (c >> i) & 1 == 0:
            u |= (1 << i)
    v = u ^ c
    product = (u * v) % MOD
    return product

# Read input
n = int(input())
a, b = map(int, input().split())

# Compute and print the result
print(maximum_product(a, b, n))