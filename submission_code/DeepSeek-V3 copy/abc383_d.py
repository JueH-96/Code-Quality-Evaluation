import math

def count_numbers_with_9_divisors(N):
    count = 0
    
    # Case 1: p^8
    p = 2
    while True:
        p8 = p ** 8
        if p8 > N:
            break
        count += 1
        p += 1
    
    # Case 2: p^2 * q^2
    # We need to find all pairs (p, q) where p < q and p^2 * q^2 <= N
    # To optimize, we can iterate p up to sqrt(sqrt(N)) and for each p, iterate q from p+1 up to sqrt(N / p^2)
    max_p = int(math.sqrt(math.sqrt(N))) + 1
    for p in range(2, max_p):
        p_sq = p * p
        if p_sq > N:
            break
        max_q = int(math.sqrt(N / p_sq)) + 1
        for q in range(p + 1, max_q):
            q_sq = q * q
            if p_sq * q_sq > N:
                break
            count += 1
    
    return count

# Read input
N = int(input())
# Compute and print the result
print(count_numbers_with_9_divisors(N))