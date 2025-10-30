# YOUR CODE HERE
import sys
import bisect

def sieve_of_eratosthenes(max_num):
    is_prime = [True] * (max_num + 1)
    p = 2
    while (p * p <= max_num):
        if (is_prime[p] == True):
            for i in range(p * p, max_num + 1, p):
                is_prime[i] = False
        p += 1
    prime_numbers = [p for p in range(2, max_num + 1) if is_prime[p]]
    return prime_numbers

def generate_400_numbers(max_value):
    primes = sieve_of_eratosthenes(10**6)
    four_hundred_numbers = set()
    
    for i in range(len(primes)):
        p = primes[i]
        p_power = p * p
        while p_power <= max_value:
            for j in range(i + 1, len(primes)):
                q = primes[j]
                q_power = q * q
                while p_power * q_power <= max_value:
                    four_hundred_numbers.add(p_power * q_power)
                    q_power *= q * q
            p_power *= p * p
    
    return sorted(four_hundred_numbers)

def main():
    input = sys.stdin.read
    data = input().split()
    
    Q = int(data[0])
    queries = list(map(int, data[1:Q+1]))
    
    max_query = max(queries)
    
    # Generate all 400 numbers up to the maximum query value
    four_hundred_numbers = generate_400_numbers(max_query)
    
    results = []
    for A in queries:
        idx = bisect.bisect_right(four_hundred_numbers, A)
        results.append(four_hundred_numbers[idx - 1])
    
    for result in results:
        print(result)