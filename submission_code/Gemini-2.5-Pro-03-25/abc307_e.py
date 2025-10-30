import sys

# Set higher recursion depth for safety, although not strictly needed for this iterative solution
# sys.setrecursionlimit(2000) 

# Function to compute the solution
def solve():
    # Read N and M from standard input
    # N is the number of people (vertices in the cycle graph)
    # M is the number of integers (colors) available, from 0 to M-1
    N, M = map(int, sys.stdin.readline().split())
    
    # Define the modulus
    MOD = 998244353

    # The problem asks for the number of ways to assign integers (colors) to N people
    # in a circle such that no two adjacent people have the same integer.
    # This is equivalent to finding the chromatic polynomial of a cycle graph C_N evaluated at M.
    # The chromatic polynomial of a cycle graph C_N is P(C_N, M) = (M-1)^N + (-1)^N * (M-1).
    # We need to compute this value modulo MOD.

    # Let K = M-1. The formula becomes K^N + (-1)^N * K mod MOD.
    
    # Calculate K = M-1 mod MOD
    # Since the problem constraints state M >= 2, M-1 >= 1.
    # Thus K = (M-1) % MOD will compute a value in the range [0, MOD-1].
    # If M-1 is a multiple of MOD, K will be 0.
    K = (M - 1) % MOD
    
    # Calculate the first term: K^N mod MOD using modular exponentiation.
    # Python's built-in pow(base, exponent, modulus) is efficient for this.
    term1 = pow(K, N, MOD)
    
    # Calculate the second term: (-1)^N * K mod MOD
    # The value of (-1)^N depends on the parity of N.
    if N % 2 == 0:
        # If N is even, (-1)^N = 1. The second term is K mod MOD.
        term2 = K
    else:
        # If N is odd, (-1)^N = -1. The second term is -K mod MOD.
        # To compute -K mod MOD and ensure the result is non-negative,
        # we can calculate (MOD - K) % MOD.
        # For example, if K=2 and MOD=10, -K mod MOD = -2 mod 10 = 8.
        # (MOD - K) % MOD = (10 - 2) % 10 = 8 % 10 = 8.
        term2 = (MOD - K) % MOD 
        
    # The final answer is the sum of the two terms modulo MOD.
    # (term1 + term2) might exceed MOD, so we take the result modulo MOD.
    # The result is guaranteed to be in the range [0, MOD-1].
    ans = (term1 + term2) % MOD
    
    # Print the final answer to standard output.
    print(ans)

# Call the solve function to execute the logic when the script runs.
solve()