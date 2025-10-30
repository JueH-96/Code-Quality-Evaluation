import sys
import math

def sieve(n):
    """ Returns a list of primes up to n, inclusive. """
    is_prime = [True] * (n + 1)
    p = 2
    while (p * p <= n):
        if (is_prime[p] == True):
            for i in range(p * p, n + 1, p):
                is_prime[i] = False
        p += 1
    primes = [p for p in range(2, n + 1) if is_prime[p]]
    return primes

def generate_400_numbers(max_limit):
    """ Generate all 400 numbers up to max_limit. """
    # We need primes up to sqrt(max_limit) because p^2 is the smallest power
    # that satisfies the even division condition.
    limit_for_primes = int(math.sqrt(max_limit)) + 1
    primes = sieve(limit_for_primes)
    
    # Set to store 400 numbers
    four_hundred_numbers = set()
    
    # Generate numbers of the form p^2 * q^2 where p and q are distinct primes
    for i in range(len(primes)):
        p = primes[i]
        p_squared = p * p
        if p_squared > max_limit:
            break
        for j in range(i + 1, len(primes)):
            q = primes[j]
            q_squared = q * q
            product = p_squared * q_squared
            if product > max_limit:
                break
            four_hundred_numbers.add(product)
    
    # Convert to sorted list
    sorted_four_hundred_numbers = sorted(four_hundred_numbers)
    return sorted_four_hundred_numbers

def main():
    input = sys.stdin.read
    data = input().split()
    Q = int(data[0])
    queries = list(map(int, data[1:]))
    
    max_query = max(queries)
    
    # Precompute all 400 numbers up to the maximum query value
    four_hundred_numbers = generate_400_numbers(max_query)
    
    # For each query, find the largest 400 number not exceeding A
    results = []
    for A in queries:
        # Binary search for the largest 400 number <= A
        lo, hi = 0, len(four_hundred_numbers) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if four_hundred_numbers[mid] <= A:
                lo = mid + 1
            else:
                hi = mid - 1
        # hi should be the index of the largest 400 number <= A
        results.append(four_hundred_numbers[hi])
    
    # Print results
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == "__main__":
    main()