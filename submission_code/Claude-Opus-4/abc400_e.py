# YOUR CODE HERE
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def get_primes_up_to(limit):
    primes = []
    for i in range(2, limit + 1):
        if is_prime(i):
            primes.append(i)
    return primes

def find_largest_400_number(A):
    # We need p^(2a) * q^(2b) <= A
    # The smallest 400 number is 2^2 * 3^2 = 36
    
    # Get primes up to sqrt(A) since p^2 <= A
    limit = int(A**0.5) + 1
    primes = get_primes_up_to(min(limit, 1000000))  # Limit prime generation
    
    best = 36  # The smallest 400 number
    
    # Try all pairs of distinct primes
    for i in range(len(primes)):
        p = primes[i]
        if p * p > A:
            break
            
        for j in range(len(primes)):
            if i == j:
                continue
            q = primes[j]
            
            # Try different powers
            p_power = p * p
            while p_power <= A:
                q_power = q * q
                while p_power * q_power <= A:
                    best = max(best, p_power * q_power)
                    q_power *= q * q
                p_power *= p * p
    
    return best

Q = int(input())
for _ in range(Q):
    A = int(input())
    print(find_largest_400_number(A))