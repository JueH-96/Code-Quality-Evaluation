import sys

def sum_of_exponents(n):
    """Calculates the sum of exponents in the prime factorization of n."""
    # Base case for n=1, although problem constraints state A_i >= 2 initially.
    # A number can become 1 during the game. S(1) = 0.
    if n == 1:
        return 0
    
    s = 0
    temp_n = n
    
    # Count exponent of 2
    if temp_n % 2 == 0:
        count = 0
        while temp_n % 2 == 0:
            count += 1
            temp_n //= 2
        s += count
        
    # Count exponents of odd primes starting from 3
    p = 3
    # We only need to check potential prime factors up to sqrt(temp_n).
    # If after dividing by all primes up to sqrt(temp_n), temp_n is still > 1,
    # the remaining temp_n must be a prime itself.
    while p * p <= temp_n:
        if temp_n % p == 0:
            count = 0
            while temp_n % p == 0:
                count += 1
                temp_n //= p
            s += count
        p += 2 # Check next odd number
        
    # If temp_n is still > 1, the remaining temp_n is a prime factor itself with exponent 1.
    if temp_n > 1:
        s += 1
        
    return s

# Read input
N = int(sys.stdin.readline())
# Read the array A from the second line
A = list(map(int, sys.stdin.readline().split()))

# Calculate the XOR sum of Grundy values (sum of exponents) for each number
total_xor_sum = 0
for a in A:
    total_xor_sum ^= sum_of_exponents(a)

# Determine the winner based on the total XOR sum
if total_xor_sum != 0:
    print("Anna")
else:
    print("Bruno")