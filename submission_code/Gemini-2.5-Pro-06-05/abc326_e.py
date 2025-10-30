# YOUR CODE HERE
import sys

def solve():
    """
    This function reads input, solves the problem, and prints the output.
    """
    MOD = 998244353

    # Use sys.stdin.readline for faster I/O in competitive programming
    input = sys.stdin.readline
    
    try:
        N = int(input())
        # The problem statement's A_i (1-indexed) corresponds to A[i-1] (0-indexed)
        A = list(map(int, input().split()))
    except (ValueError, IndexError):
        # Graceful exit for empty input, useful for some testing environments
        return

    # Let E(k) be the expected future salary given that the current value of x is k.
    # We want to find E(0).
    #
    # A recurrence relation can be derived for E(k):
    # E(k) = (1/N) * sum_{y=k+1 to N} (A_y + E(y))
    #
    # By comparing the expressions for E(k) and E(k-1), we get a simpler recurrence:
    # E(k-1) = (A_k + (N + 1) * E(k)) * N_inv (mod MOD)
    #
    # The base case is E(N) = 0, as no roll y can be > N.
    # We compute E(N-1), E(N-2), ..., E(0) iteratively by working backwards.
    
    # Calculate the modular inverse of N.
    # This is safe because N < MOD and MOD is prime, so N is not a multiple of MOD.
    N_inv = pow(N, -1, MOD)
    
    # Pre-calculate (N + 1) for the recurrence.
    N_plus_1 = N + 1

    # e_k will represent E(k). We start with the base case E(N) = 0.
    e_k = 0

    # Iterate k downwards from N to 1. In each step, we calculate E(k-1) based on E(k).
    for k in range(N, 0, -1):
        # A_k in the formula is A[k-1] in our 0-indexed list.
        a_k = A[k - 1]
        
        # Apply the recurrence relation: E(k-1) = (a_k + (N+1) * E(k)) * N_inv
        # `e_k` currently holds E(k) from the previous state in the backward iteration.
        numerator = (a_k + (N_plus_1 * e_k)) % MOD
        e_k_minus_1 = (numerator * N_inv) % MOD
        
        # The new value becomes the E(k) for the next, smaller k.
        e_k = e_k_minus_1
    
    # After the loop completes, e_k holds E(0), which is the final answer.
    print(e_k)

solve()