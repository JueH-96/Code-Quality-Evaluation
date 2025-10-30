A, B, M = map(int, input().split())
AB = A * B

# Calculate the product of hook lengths
product = 1
for i in range(A):
    for j in range(B):
        hook = (B - (j + 1)) + (A - (i + 1)) + 1
        product *= hook

# Calculate AB! mod M
fact = 1
for k in range(1, AB + 1):
    fact = (fact * k) % M

# Compute product mod M
product_mod = product % M

# Compute modular inverse of product_mod
inv = pow(product_mod, M - 2, M)

# Multiply by B mod M
b_mod = B % M

# Calculate the answer
answer = (fact * b_mod) % M
answer = (answer * inv) % M

print(answer)