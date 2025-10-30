def factorial_mod(n, mod):
    result = 1
    for i in range(2, n + 1):
        result = (result * i) % mod
    return result

def mod_inverse(x, mod):
    # Fermat's Little Theorem: x^(mod-1) ≡ 1 (mod mod) => x^(mod-2) ≡ x^(-1) (mod mod)
    return pow(x, mod - 2, mod)

def count_permutations(A, B, M):
    AB = A * B
    fact_AB = factorial_mod(AB, M)
    
    hook_product = 1
    for i in range(A):
        for j in range(B):
            hook_length = (A - i) + (B - j) - 1
            hook_product = (hook_product * hook_length) % M
    
    hook_product_inv = mod_inverse(hook_product, M)
    
    result = (fact_AB * hook_product_inv) % M
    return result

# Read input
import sys
input = sys.stdin.read
A, B, M = map(int, input().strip().split())

# Output the result
print(count_permutations(A, B, M))