A, B, M = map(int, input().split())

n = A * B - 1

# Compute factorial of n modulo M
fact = 1
for i in range(2, n + 1):
    fact = fact * i % M

# Compute the product of hook lengths
product = 1

# Part 1: cells (i,j) where i < A and j < B
for i in range(1, A):
    for j in range(1, B):
        term = (A - i) + (B - j)
        product = product * term % M

# Part 2: cells (i, B) where i < A
for i in range(1, A):
    term = A - i
    product = product * term % M

# Part 3: cells (A, j) where j < B
for j in range(1, B):
    term = B - j
    product = product * term % M

# Compute the modular inverse of product modulo M
inv_product = pow(product, M - 2, M)

# The answer is fact multiplied by the inverse of product, modulo M
ans = fact * inv_product % M

print(ans)