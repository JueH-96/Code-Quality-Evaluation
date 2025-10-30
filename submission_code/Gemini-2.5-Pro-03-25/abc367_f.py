# YOUR CODE HERE
import sys
import random

# Function to solve the problem
def solve():
    # Read N (length of sequences) and Q (number of queries)
    N, Q = map(int, sys.stdin.readline().split())
    
    # Read sequences A and B
    # The problem statement uses 1-based indexing for sequences and queries.
    # We store the sequences with a dummy 0 element at index 0 to easily use 1-based indexing.
    A_input = list(map(int, sys.stdin.readline().split()))
    B_input = list(map(int, sys.stdin.readline().split()))
    
    # Prepend 0 to make arrays 1-indexed internally
    A = [0] + A_input
    B = [0] + B_input

    # --- Hashing Setup ---
    # The core idea is to check if two subsequences are permutations of each other
    # by comparing the hash of their multisets. Two multisets are identical if and only if
    # they contain the same elements with the same frequencies.
    # We use polynomial hashing based on summation. Each unique value `v` is assigned a random hash `h(v)`.
    # The hash of a multiset {x1, x2, ..., xk} is defined as (sum(h(xi))) mod M.
    # Using two distinct large prime moduli significantly reduces the probability of hash collisions.
    M1 = 10**9 + 7
    M2 = 998244353
    
    # Generate random hash values for each integer from 1 to N.
    # The values in sequences A and B are guaranteed to be between 1 and N.
    
    # Create arrays to store the hash values for numbers 1 to N.
    # Size N+1 is needed because values range up to N, and we use 1-based indexing.
    hash_map1 = [0] * (N + 1)
    hash_map2 = [0] * (N + 1)
    
    # Populate the hash maps with random values in the range [1, M-1].
    # Choosing values > 0 avoids issues where adding an element doesn't change the hash.
    # The random module uses a pseudo-random generator. For typical competitive programming, this is sufficient.
    # A fixed seed could be used for debugging (e.g., random.seed(42)) but is generally omitted in final submissions.
    
    for i in range(1, N + 1):
        hash_map1[i] = random.randint(1, M1 - 1)
        hash_map2[i] = random.randint(1, M2 - 1)

    # --- Precomputation of Prefix Hash Sums ---
    # To efficiently calculate the hash sum of any subarray/subsequence range in O(1) time,
    # we precompute prefix hash sums. Prefix sum P[k] stores the sum of hashes for elements from index 1 to k.
    # The hash sum for a range [l, r] is then P[r] - P[l-1].
    
    # Compute prefix hash sums for sequence A using both hash functions.
    prefix_hash_A1 = [0] * (N + 1)
    prefix_hash_A2 = [0] * (N + 1)
    for k in range(1, N + 1):
        # Get the integer value at position k in A (1-based index)
        val = A[k] 
        
        # Update prefix sums using the generated hash values and moduli.
        # P[k] = (P[k-1] + hash(value at index k)) mod M
        prefix_hash_A1[k] = (prefix_hash_A1[k-1] + hash_map1[val]) % M1
        prefix_hash_A2[k] = (prefix_hash_A2[k-1] + hash_map2[val]) % M2

    # Compute prefix hash sums for sequence B using both hash functions.
    prefix_hash_B1 = [0] * (N + 1)
    prefix_hash_B2 = [0] * (N + 1)
    for k in range(1, N + 1):
        # Get the integer value at position k in B (1-based index)
        val = B[k]

        # Update prefix sums similarly for B.
        prefix_hash_B1[k] = (prefix_hash_B1[k-1] + hash_map1[val]) % M1
        prefix_hash_B2[k] = (prefix_hash_B2[k-1] + hash_map2[val]) % M2

    # --- Process Queries ---
    results = [] # Store answers to queries to print all at once at the end for efficiency
    for _ in range(Q):
        # Read query parameters (l, r for A; L, R for B), all are 1-based indices.
        l, r, L, R = map(int, sys.stdin.readline().split())
        
        # Condition 1: Check if lengths match.
        # The length of subsequence A[l..r] is r - l + 1.
        # The length of subsequence B[L..R] is R - L + 1.
        # If the lengths differ, they cannot be permutations of each other.
        if (r - l + 1) != (R - L + 1):
            results.append("No")
            continue

        # Condition 2: Check if the multisets of elements match using hashing.
        # Calculate hash sum for subsequence A[l..r].
        # Hash sum H(A[l..r]) = (prefix_hash[r] - prefix_hash[l-1]) mod M
        # Modular subtraction needs care: (a - b + M) % M ensures the result is non-negative.
        hash_A1 = (prefix_hash_A1[r] - prefix_hash_A1[l-1] + M1) % M1
        hash_A2 = (prefix_hash_A2[r] - prefix_hash_A2[l-1] + M2) % M2

        # Calculate hash sum for subsequence B[L..R].
        hash_B1 = (prefix_hash_B1[R] - prefix_hash_B1[L-1] + M1) % M1
        hash_B2 = (prefix_hash_B2[R] - prefix_hash_B2[L-1] + M2) % M2

        # Compare the computed hashes. If both pairs of hashes match, we conclude
        # with high probability that the multisets are identical.
        if hash_A1 == hash_B1 and hash_A2 == hash_B2:
            results.append("Yes")
        else:
            # If any pair of hashes differs, the multisets are definitely different.
            results.append("No")

    # Print all collected results, each on a new line.
    # Using '
'.join is generally faster than multiple print calls for a large number of results.
    print('
'.join(results))

# Ensure the main part of the script calls the solve function when executed directly.
if __name__ == '__main__':
    solve()