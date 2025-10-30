def count_divisors(n):
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            if i * i == n:
                count += 1
            else:
                count += 2
        i += 1
    return count

def solve(N):
    # Numbers with 9 divisors must be in one of these forms:
    # 1. p^8 where p is prime
    # 2. p^2 * q^2 where p,q are distinct primes
    # 3. p^4 * q where p,q are distinct primes
    
    count = 0
    
    # Check p^8 form
    i = 2
    while i**8 <= N:
        count += 1
        i += 1
    
    # Check p^2 * q^2 form
    for p in range(2, int(N**0.25) + 1):
        if count_divisors(p) == 2:  # p is prime
            for q in range(p + 1, int((N//(p*p))**0.5) + 1):
                if count_divisors(q) == 2:  # q is prime
                    if p*p*q*q <= N:
                        count += 1
    
    # Check p^4 * q form
    for p in range(2, int(N**0.2) + 1):
        if count_divisors(p) == 2:  # p is prime
            p4 = p**4
            for q in range(2, int(N/p4) + 1):
                if count_divisors(q) == 2:  # q is prime
                    if p4*q <= N:
                        count += 1
    
    return count

# Read input
N = int(input())
print(solve(N))