# YOUR CODE HERE
import sys

# Set higher recursion depth for safety, although not strictly needed for this iterative approach
# sys.setrecursionlimit(200000) # Commented out as it is not needed for iterative approach

MOD = 998244353

def solve():
    # Read N and M from standard input
    N, M = map(int, sys.stdin.readline().split())

    # Constraints state N >= 1, M >= 1.
    # The formula derived and verified against base cases and sample is E = M^2 * (2*N * H_N - 1) mod P
    # where H_N is the N-th Harmonic number H_N = sum_{k=1}^N 1/k.
    # All calculations are performed modulo MOD = 998244353.

    # We need to calculate H_N mod MOD. This requires modular inverses k^{-1} mod MOD for k=1..N.
    # We can compute these inverses efficiently in linear time.

    # Calculate modular inverses for 1 to N using linear time algorithm O(N)
    # inv[i] stores the modular inverse of i modulo MOD.
    inv = [0] * (N + 1)
    
    # Base case for inverse calculation: inv[1] = 1
    if N >= 1:
      inv[1] = 1
      
    # Compute inverses iteratively for i = 2 to N
    # This algorithm relies on MOD being prime. 998244353 is prime.
    for i in range(2, N + 1):
        # The formula inv[i] = -(MOD // i) * inv[MOD % i] % MOD is derived from
        # MOD = q*i + r  =>  q*i + r = 0 (mod MOD)
        # Multiply by inv[i] * inv[r]: q * inv[r] + inv[i] = 0 (mod MOD)
        # inv[i] = -q * inv[r] (mod MOD)
        # where q = MOD // i and r = MOD % i.
        # Using (MOD - ...) ensures the result is in the range [0, MOD-1].
        inv[i] = (MOD - (MOD // i) * inv[MOD % i]) % MOD 


    # Calculate Harmonic Number H_N mod P using the precomputed inverses O(N)
    # H_N = sum_{k=1}^N k^{-1} mod MOD
    current_H_N = 0
    for k in range(1, N + 1):
        current_H_N = (current_H_N + inv[k]) % MOD
    
    # Now we have H_N mod MOD stored in current_H_N.

    # Compute M^2 mod P
    # Python handles large integers automatically, so direct calculation M*M is fine.
    # Take modulo at the end.
    M_squared = (M * M) % MOD

    # Compute 2*N mod P
    Two_N = (2 * N) % MOD

    # Compute the final expected value E = M^2 * (2*N * H_N - 1) mod P
    # All calculations are done modulo MOD.
    
    # Calculate (2*N * H_N) mod P
    term1 = (Two_N * current_H_N) % MOD 
    
    # Calculate (2*N * H_N - 1) mod P
    # Use the standard way to handle subtraction in modular arithmetic: (a - b + MOD) % MOD
    # to ensure the result remains non-negative.
    term_in_paren = (term1 - 1 + MOD) % MOD 
    
    # Final multiplication: M^2 * (term_in_paren) mod P
    result = (M_squared * term_in_paren) % MOD

    # Print the final result
    print(result)

# Call the solve function to execute the logic
solve()

# END OF YOUR CODE HERE