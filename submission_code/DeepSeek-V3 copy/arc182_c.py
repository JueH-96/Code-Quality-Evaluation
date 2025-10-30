MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    
    # Precompute the number of divisors for each possible product
    # Since M is small (up to 16), we can precompute all possible products and their divisors
    # However, for large N, we need a smarter approach
    
    # The key insight is that the number of divisors of the product of a sequence is determined by the exponents of the prime factors in the product
    # So, we need to count the number of sequences that result in each possible combination of prime exponents
    
    # First, factorize all numbers from 1 to M
    # For each number, represent it as a vector of exponents of the primes
    
    # The primes up to 16 are 2, 3, 5, 7, 11, 13
    primes = [2, 3, 5, 7, 11, 13]
    num_primes = len(primes)
    
    # For each number from 1 to M, represent it as a vector of exponents
    factor_vectors = []
    for x in range(1, M+1):
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while x % p == 0:
                vec[i] += 1
                x = x // p
        factor_vectors.append(vec)
    
    # Now, for each sequence, the product's exponent vector is the sum of the vectors of its elements
    # The number of divisors of the product is the product of (exponent + 1) for each prime
    
    # To compute the total sum of scores, we need to consider all possible sequences and their corresponding exponent vectors
    
    # Since N can be up to 1e18, we need a way to compute this efficiently
    
    # The key is to use generating functions or dynamic programming to count the number of sequences that result in each possible exponent vector
    
    # However, given the small size of M and the number of primes, we can precompute all possible exponent vectors and their counts
    
    # The maximum exponent for each prime is N * (maximum exponent in factor_vectors)
    # For M=16, the maximum exponent for any prime is 4 (for 16 = 2^4)
    # So, the maximum exponent for any prime in the product is 4 * N
    
    # But since N can be up to 1e18, this is not feasible
    
    # Instead, we need to find a way to compute the sum of the number of divisors for all sequences without explicitly enumerating all sequences
    
    # The number of divisors of the product of a sequence is the product of (e_i + 1) where e_i is the exponent of the i-th prime in the product
    
    # So, the total sum of scores is the sum over all sequences of the product of (e_i + 1) where e_i is the sum of the exponents of the i-th prime in the sequence
    
    # This can be rewritten as the product over all primes of the sum over all sequences of (e_i + 1) where e_i is the sum of the exponents of the i-th prime in the sequence
    
    # Wait, no. The sum of the product is not the product of the sums
    
    # Instead, we need to consider the generating function approach
    
    # For each prime, the contribution to the number of divisors is (e_i + 1), where e_i is the sum of the exponents of the i-th prime in the sequence
    
    # So, for each prime, we need to compute the sum over all sequences of (e_i + 1)
    
    # This can be done by considering the generating function for each prime
    
    # For each prime p, the generating function for the exponent e_i is:
    # G_p(x) = \prod_{k=1}^M (1 + x^{a_{k,p}} + x^{2a_{k,p}} + ... + x^{N a_{k,p}})
    # where a_{k,p} is the exponent of p in the k-th number
    
    # Then, the sum of (e_i + 1) over all sequences is the coefficient of x^0 in G_p(x) * (1 + 2x + 3x^2 + ... + (N+1)x^N)
    
    # However, this is still computationally infeasible for large N
    
    # Instead, we can use the fact that the sum of (e_i + 1) over all sequences is equal to the sum over all sequences of e_i plus the number of sequences
    
    # The number of sequences is \sum_{k=1}^N M^k = M (M^N - 1) / (M - 1) if M != 1, else N
    
    # The sum of e_i over all sequences is the sum over all sequences of the sum of the exponents of p in the sequence
    
    # This is equal to the sum over all elements in all sequences of the exponent of p in that element
    
    # The number of sequences of length k is M^k, and each element appears in M^{k-1} sequences of length k
    
    # So, the total sum of the exponents of p over all sequences is the sum over all elements of (exponent of p in the element) * (sum over k=1 to N of M^{k-1})
    
    # The sum over k=1 to N of M^{k-1} is (M^N - 1) / (M - 1) if M != 1, else N
    
    # So, the sum of e_i over all sequences is (sum over all elements of (exponent of p in the element)) * (M^N - 1) / (M - 1) if M != 1, else N * (sum over all elements of (exponent of p in the element))
    
    # Therefore, the sum of (e_i + 1) over all sequences is (sum of e_i) + (number of sequences)
    
    # So, for each prime p, the sum of (e_i + 1) over all sequences is:
    # (sum over all elements of (exponent of p in the element)) * (M^N - 1) / (M - 1) + (M (M^N - 1) / (M - 1)) if M != 1, else N * (sum over all elements of (exponent of p in the element)) + N
    
    # Then, the total sum of scores is the product over all primes of this value
    
    # Now, we can compute this efficiently
    
    # First, compute the number of sequences
    if M == 1:
        num_sequences = N
    else:
        num_sequences = (pow(M, N, MOD * (M - 1)) - 1) // (M - 1)
        num_sequences %= MOD
    
    # Compute the sum of exponents for each prime
    sum_exponents = [0] * num_primes
    for vec in factor_vectors:
        for i in range(num_primes):
            sum_exponents[i] += vec[i]
    
    # Compute the sum of (e_i + 1) for each prime
    sum_e_plus_1 = []
    for i in range(num_primes):
        if M == 1:
            s = N * sum_exponents[i] + N
        else:
            s = sum_exponents[i] * ((pow(M, N, MOD * (M - 1)) - 1) // (M - 1)) + num_sequences
        s %= MOD
        sum_e_plus_1.append(s)
    
    # Compute the product of all sum_e_plus_1
    total = 1
    for s in sum_e_plus_1:
        total = (total * s) % MOD
    
    print(total)

if __name__ == "__main__":
    main()