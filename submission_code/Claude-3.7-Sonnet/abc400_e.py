def generate_primes(n):
    """Generate all primes up to n using the Sieve of Eratosthenes."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(n + 1) if sieve[i]]

def largest_400_not_exceeding(A, primes):
    """Find the largest 400 number not exceeding A."""
    max_400 = 0
    
    # We only need to consider primes up to sqrt(A)
    max_prime_index = 0
    while max_prime_index < len(primes) and primes[max_prime_index] <= int(A ** 0.5):
        max_prime_index += 1
    
    for i in range(max_prime_index):
        p = primes[i]
        p_power = p * p  # Starting with p^2
        while p_power <= A:
            for j in range(i + 1, max_prime_index):
                q = primes[j]
                if p_power * q * q > A:
                    break
                q_power = q * q  # Starting with q^2
                while p_power * q_power <= A:
                    max_400 = max(max_400, p_power * q_power)
                    q_power *= q * q  # Multiply by q^2
                    if q_power > A / p_power:
                        break
            p_power *= p * p  # Multiply by p^2
            if p_power > A:
                break
    
    return max_400

def main():
    # For a 400 number not exceeding 10^12, prime factors can be up to 5 * 10^5
    max_possible_prime = 5 * 10**5
    primes = generate_primes(max_possible_prime)
    
    Q = int(input())
    for _ in range(Q):
        A = int(input())
        print(largest_400_not_exceeding(A, primes))

if __name__ == "__main__":
    main()