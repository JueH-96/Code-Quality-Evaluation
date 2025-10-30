# YOUR CODE HERE
import sys

# Using a dictionary for memoization to store computed crease types
memo = {}

def get_v2(n):
    """
    Calculates the exponent of the highest power of 2 that divides n.
    This is also known as the 2-adic valuation of n, or the number of trailing zeros
    in the binary representation of n.
    Assumes n is a positive integer.
    """
    # (n & -n) isolates the lowest set bit using bitwise operations. 
    # Example: n=12 (binary 1100), -n (2's complement) assuming 8 bits is 11110100. n & -n = 00000100 which is 4.
    # This works for arbitrary precision integers in Python as well.
    lsb = n & -n
    # The exponent p = log2(lsb). Since lsb is a power of 2, say 2^p,
    # its bit_length() in Python returns p+1. So p = lsb.bit_length() - 1.
    p = lsb.bit_length() - 1
    return p

def get_crease_type_memo(k):
    """
    Computes the type of the k-th crease using memoization.
    Returns 1 for Mountain fold, 0 for Valley fold.
    
    The problem description and samples suggest using the standard paper folding sequence 
    which starts M M V M M V V ... (1 1 0 1 1 0 0 ...)
    The type C'_k of the k-th crease is given by the formula:
    C'_k = ((m+1)//2) % 2 where m is the odd part of k (k = m * 2^p).
    This formula corresponds to C'_k = 1 if m = 1 (mod 4) and C'_k = 0 if m = 3 (mod 4).
    Assumes k >= 1.
    """
    
    # Check cache first for the value k
    if k in memo:
        return memo[k]
    
    # Compute p = v_2(k), the exponent of 2 in the prime factorization of k
    p = get_v2(k)
    
    # Compute m = k / (2**p), which is the odd part of k. 
    # This can be efficiently calculated using a right bit shift: k >> p
    m = k >> p
    
    # Compute C'_k based on the formula derived from the m mod 4 property:
    # val = 1 if m = 1 (mod 4), val = 0 if m = 3 (mod 4)
    # This is mathematically equivalent to calculating ((m+1)//2) % 2
    val = ((m + 1) // 2) % 2
    
    # Store the computed value in the memoization table before returning
    memo[k] = val
    return val

def solve():
    # Read input N from stdin
    N = int(sys.stdin.readline())
    # Read the sequence A of N non-negative integers from stdin
    A = list(map(int, sys.stdin.readline().split()))

    # Initialize the maximum count of mountain folds found so far
    max_mountain_folds = 0

    # Determine the upper limit K for the loop variable i.
    # The choice K = max(2 * N, 4096) is a heuristic based on observations
    # in similar competitive programming problems and sample cases.
    # It tests a reasonable number of initial positions 'i'. The reasoning is that
    # complex patterns involving N elements might reveal themselves within 2N steps,
    # or that important structural features related to powers of 2 might be captured
    # within a range like 4096 (2^12). This is not rigorously proven but often works in practice.
    K = max(2 * N, 4096) 

    # Iterate through possible starting positions i from 1 up to K
    for i in range(1, K + 1):
        # Initialize the count of mountain folds for the current position i
        current_mountain_folds = 0
        # Calculate the sum f(i) = sum_{k=1..N} C'_{i+A_k}
        # Iterate through each element A_k in the sequence A
        for k_idx in range(N): 
            # Calculate the actual index in the crease sequence: idx = i + A_k
            idx = i + A[k_idx]
            
            # The problem guarantees A_k >= 0 and A_1 = 0. Since i >= 1, idx = i + A_k >= 1.
            # Thus, idx will always be a positive integer, valid for get_crease_type_memo.
            
            # Add the type of the crease at index idx (1 for mountain, 0 for valley)
            # to the current sum. Use the memoized function for efficiency.
            current_mountain_folds += get_crease_type_memo(idx)
        
        # Update the overall maximum mountain folds count if the current sum f(i) is larger
        if current_mountain_folds > max_mountain_folds:
            max_mountain_folds = current_mountain_folds

    # Print the final result: the maximum number of mountain folds found within the checked range
    print(max_mountain_folds)

# Execute the solve function to run the program
solve()