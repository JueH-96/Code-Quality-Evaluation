# YOUR CODE HERE
import sys

# Set recursion depth limit if needed, likely not necessary for this problem
# sys.setrecursionlimit(2000)

def solve():
    # Read input
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Maximum possible value of A_i according to constraints
    MAX_A_CONSTRAINT = 2 * 10**5

    # Precompute smallest prime factor using sieve up to MAX_A_CONSTRAINT
    # spf[i] will store the smallest prime factor of i
    # We need spf up to MAX_A_CONSTRAINT to factorize any number up to MAX_A_CONSTRAINT
    spf = list(range(MAX_A_CONSTRAINT + 1))
    # Sieve starts from 2
    for i in range(2, int(MAX_A_CONSTRAINT**0.5) + 1):
        # If i is still marked as i, it means i is prime
        if spf[i] == i:
            # Mark multiples of i starting from i*i
            # Multiples smaller than i*i would have been marked by a smaller prime
            for j in range(i * i, MAX_A_CONSTRAINT + 1, i):
                if spf[j] == j: # Only update if not already marked by a smaller prime
                    spf[j] = i

    # Function to get the square-free part of a number n > 0
    # The square-free part is the product of prime factors that appear
    # with an odd exponent in the prime factorization of n.
    def get_square_free_part(n):
        # This function is intended for n > 0.
        # If n == 0, its square-free part is not well-defined in this context.
        # The main logic handles n=0 separately.
        if n <= 0:
             # This branch should ideally not be reached when called from the main logic
             # as we only call it for x > 0 from the input array A.
             # Returning 1 or raising an error could be options, but it's better
             # to ensure the caller provides n > 0.
             # For safety, let's assume any non-positive input should behave such
             # that its product with itself is a square (like 0*0=0).
             # However, the problem defines square-free part for positive integers.
             # We rely on the main logic to not call this with n=0.
             # Based on how it's used, this block is unreachable with n=0.
             # If n=1, the loop condition `while temp_n > 1` is false, and res=1 is returned, which is correct.
             pass # The main logic only calls this for x > 0

        res = 1
        temp_n = n
        # Use spf to find prime factors efficiently
        # The loop continues until temp_n becomes 1, meaning all prime factors are processed
        while temp_n > 1:
            # Get the smallest prime factor of the current temp_n
            p = spf[temp_n]
            count = 0
            # Count the exponent of the prime factor p in temp_n
            while temp_n % p == 0:
                count += 1
                temp_n //= p # Divide temp_n by p

            # If the exponent of prime p is odd, include this prime in the square-free part
            if count % 2 == 1:
                res *= p
        return res

    c0 = 0 # Count of zeros in array A
    sf_counts = {} # Dictionary to store counts of each square-free part among non-zero numbers

    # Iterate through the input array A
    for x in A:
        if x == 0:
            c0 += 1
        else:
            # For non-zero elements, calculate their square-free part
            sf = get_square_free_part(x)
            # Store the count of this square-free part
            sf_counts[sf] = sf_counts.get(sf, 0) + 1

    # Calculate the total number of pairs (i, j) with 1 <= i < j <= N
    # This is N choose 2.
    total_possible_pairs = N * (N - 1) // 2

    # Calculate the number of pairs (i, j) with 1 <= i < j <= N where both A[i] > 0 and A[j] > 0
    num_non_zeros = N - c0
    # This is (N - c0) choose 2. If num_non_zeros < 2, this correctly evaluates to 0.
    pairs_both_non_zero = num_non_zeros * (num_non_zeros - 1) // 2

    # The number of pairs (i, j) with 1 <= i < j <= N where at least one of A[i] or A[j] is 0.
    # For any such pair, A[i]*A[j] = 0, which is 0^2, a square number.
    # So, all these pairs satisfy the condition.
    # This count is the total possible pairs minus the pairs where both elements are non-zero.
    valid_pairs = total_possible_pairs - pairs_both_non_zero

    # Now, we need to count pairs (i, j) with 1 <= i < j <= N where both A[i] > 0 and A[j] > 0
    # and A[i]*A[j] is a square number.
    # This happens if and only if get_square_free_part(A[i]) == get_square_free_part(A[j]).
    # We grouped the non-zero elements by their square-free part in the sf_counts dictionary.
    # For each distinct square-free part 's', let its frequency among non-zero elements be 'count'.
    # Any pair of these 'count' elements will form a valid pair (i, j) where A[i]>0, A[j]>0,
    # their original indices satisfy i < j (implicitly covered by choosing pairs from a set),
    # and A[i]*A[j] is a square because their square-free parts are equal.
    # The number of such pairs for a specific square-free part 's' is the number of ways
    # to choose 2 elements from the 'count' elements having that square-free part, which is count * (count - 1) // 2.
    for count in sf_counts.values():
        valid_pairs += count * (count - 1) // 2

    # Print the final result
    print(valid_pairs)

# Execute the solve function
solve()