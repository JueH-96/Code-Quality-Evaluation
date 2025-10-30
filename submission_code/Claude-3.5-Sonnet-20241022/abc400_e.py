def get_prime_factors(n):
    factors = {}
    d = 2
    while n > 1:
        count = 0
        while n % d == 0:
            count += 1
            n //= d
        if count > 0:
            factors[d] = count
        d += 1 if d == 2 else 2
        if d * d > n:
            if n > 1:
                factors[n] = 1
            break
    return factors

def is_400_number(n):
    # Get prime factorization
    factors = get_prime_factors(n)
    
    # Check if number has exactly 2 distinct prime factors
    if len(factors) != 2:
        return False
    
    # Check if each prime factor appears even number of times
    for p, count in factors.items():
        if count % 2 != 0:
            return False
            
    return True

def find_largest_400_number(A):
    # Special cases for small numbers
    if A <= 36:
        return 36  # 36 = 2^2 * 3^2 is the smallest 400 number
        
    # Start from A and go down until we find a 400 number
    n = A
    while n >= 36:
        if is_400_number(n):
            return n
        n -= 1

# Read input
Q = int(input())
for _ in range(Q):
    A = int(input())
    result = find_largest_400_number(A)
    print(result)