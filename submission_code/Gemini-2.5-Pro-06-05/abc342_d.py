import sys
from collections import Counter

def main():
    """
    This function reads the input, solves the problem, and prints the output.
    """
    # Use fast I/O
    input = sys.stdin.readline
    
    try:
        N_str = input()
        if not N_str: return
        N = int(N_str)
        A = list(map(int, input().split()))
    except (IOError, ValueError):
        # Gracefully handle empty or malformed input, though not expected on judging platforms.
        return

    # The maximum possible value for an element in A, from the problem constraints.
    MAX_VAL = 200000

    # Precompute the Smallest Prime Factor (SPF) for all numbers up to MAX_VAL using a sieve.
    # This allows for very fast factorization later.
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:  # This condition means 'i' is a prime number.
            # Mark the SPF for all multiples of 'i'.
            for j in range(i * i, MAX_VAL + 1, i):
                # Update spf[j] only if it hasn't been set by a smaller prime.
                if spf[j] == j:
                    spf[j] = i

    def get_square_free_part(n):
        """
        Calculates the square-free part of a positive integer 'n' using the precomputed SPF table.
        The square-free part is the product of prime factors that have an odd exponent.
        """
        res = 1
        # Repeatedly divide 'n' by its smallest prime factor.
        while n > 1:
            p = spf[n]
            count = 0
            while n % p == 0:
                count += 1
                n //= p
            # If the prime factor 'p' appeared an odd number of times, include it in the result.
            if count % 2 == 1:
                res *= p
        return res

    # --- Main Problem-Solving Logic ---

    # Step 1: Handle pairs involving zero.
    # A_i * A_j is 0 (a perfect square) if A_i=0 or A_j=0.
    count_zero = A.count(0)
    
    # Calculate pairs of two zeros: C(count_zero, 2)
    ans = count_zero * (count_zero - 1) // 2
    # Add pairs of one zero and one non-zero: count_zero * (N - count_zero)
    ans += count_zero * (N - count_zero)

    # Step 2: Handle pairs of non-zero numbers.
    # A_i * A_j is a square iff they have the same square-free part.
    non_zeros = [x for x in A if x > 0]
    
    if non_zeros:
        # Compute the square-free part for each non-zero number.
        sf_parts = [get_square_free_part(x) for x in non_zeros]
        
        # Count frequencies of each unique square-free part.
        counts = Counter(sf_parts)
        
        # For each group of numbers with the same square-free part, add the number of pairs.
        # If a part appears 'c' times, it forms C(c, 2) pairs.
        for c in counts.values():
            ans += c * (c - 1) // 2

    # Print the total number of valid pairs.
    print(ans)

if __name__ == "__main__":
    main()