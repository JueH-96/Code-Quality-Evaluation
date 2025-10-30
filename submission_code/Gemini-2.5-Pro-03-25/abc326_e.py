# YOUR CODE HERE
import sys

# Set higher recursion depth for safety, although this problem uses iteration. 
# It's generally good practice in competitive programming for Python.
# sys.setrecursionlimit(2000000) 

# Define the modulus
P = 998244353

def fast_pow(base, power):
    """
    Computes (base^power) % P efficiently using binary exponentiation.
    P is the global modulus.
    """
    result = 1
    # Ensure base is within the modulo ring initially to prevent potential overflow
    # if base is very large, although here base is N which is constrained.
    base %= P 
    while power > 0:
        # If power is odd, multiply result with base
        if power % 2 == 1:
            result = (result * base) % P
        # Square the base and halve the power
        base = (base * base) % P
        power //= 2
    return result

def inverse(a):
    """
    Computes the modular multiplicative inverse of a under P.
    Assumes P is a prime number and 'a' is not divisible by P.
    Uses Fermat's Little Theorem: a^(P-2) = a^(-1) mod P for prime P.
    """
    # Constraints ensure N >= 1 and N < P, so N is not divisible by P.
    return fast_pow(a, P - 2)

def solve():
    # Read N from input
    N = int(sys.stdin.readline())
    # Read the sequence A_1, ..., A_N from input
    A = list(map(int, sys.stdin.readline().split()))
    
    # Adjust A to be 1-based indexing for convenience.
    # A_adj[k] will store the value A_k from the problem statement.
    # We create a list of size N+1, index 0 unused.
    A_adj = [0] * (N + 1)
    for i in range(N):
        # The constraints state 0 <= A_i < P, so A[i] is already in the correct range.
        # No initial modulo P needed for A_i values.
        A_adj[i+1] = A[i] 

    # Compute the modular inverse of N modulo P.
    # Needed for the recurrence relation calculation.
    N_inv = inverse(N)

    # Initialize an array E to store the expected values.
    # E[k] will store the expected future salary given the current state x=k.
    # The size is N+1 to accommodate states 0 through N.
    E = [0] * (N + 1)
    
    # Base case for the recurrence: E[N] = 0.
    # If the current maximum value seen is N, any subsequent roll y (1 <= y <= N)
    # will satisfy N >= y, terminating the process. So the expected future salary is 0.
    # E is already initialized with zeros, so E[N] is correctly set to 0.
    
    # Precompute (N+1) mod P. This is used repeatedly in the loop.
    N_plus_1 = (N + 1) % P
    
    # We derive the recurrence relation: E_{k-1} = (A_k + (N+1) E_k) * N_inv mod P.
    # To compute E[0], we need E[1], which needs E[2], ..., up to E[N].
    # So we iterate k from N down to 1, computing E[k-1] based on E[k].
    
    for k in range(N, 0, -1):
        # Get the value A_k for the current step k.
        current_A = A_adj[k]
        
        # Calculate the term (N+1) * E[k] mod P.
        # E[k] holds the expected future salary starting from state k.
        # This value was computed in the previous iteration (k+1 -> k) or is the base case E[N]=0.
        # All values are kept within [0, P-1] due to modulo operations.
        term2 = (N_plus_1 * E[k]) % P
        
        # Calculate the numerator: A_k + (N+1) * E[k] mod P.
        # Both current_A (A_k) and term2 are non-negative and < P.
        numerator = (current_A + term2) % P
        
        # Calculate E[k-1] using the recurrence relation.
        # Multiply the numerator by the modular inverse of N.
        E[k-1] = (numerator * N_inv) % P

    # The final answer is E[0], which represents the expected total salary
    # starting from the initial state x=0.
    print(E[0])

# Execute the solve function to run the program logic.
solve()