# YOUR CODE HERE
import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353

    # Calculate N_inv = N^(MOD-2) % MOD
    # This is the modular multiplicative inverse of N modulo MOD.
    # Fermat's Little Theorem applies because MOD is prime.
    N_inv = pow(N, MOD - 2, MOD)

    # E[k] will store the expected total salary received if the current value 'x' is k.
    # We need to find E[0].
    # The array A is 0-indexed (A[0] corresponds to A_1, A[N-1] to A_N).
    # E has size N+1 to store E_0, E_1, ..., E_N.
    E = [0] * (N + 1)

    # Base case: If x = N, the process always terminates immediately.
    # So, the expected additional salary from state N is 0.
    E[N] = 0

    # `current_sum` will store sum_{y=x+1 to N} (A_y + E_y)
    # When initialized before the loop, it represents S_N (sum from N+1 to N), which is 0.
    current_sum = 0 

    # Iterate x from N-1 down to 0
    # In each iteration for 'x', we calculate E[x].
    for x in range(N - 1, -1, -1):
        # The sum S_x = (A_{x+1} + E_{x+1}) + S_{x+1}
        # A_{x+1} is A[x] in the 0-indexed input array.
        # E_{x+1} is E[x+1] (which has already been calculated in previous iterations).
        
        # Calculate the term to add: (A_{x+1} + E_{x+1})
        term_to_add = (A[x] + E[x+1]) % MOD
        
        # Update current_sum from S_{x+1} to S_x.
        current_sum = (current_sum + term_to_add) % MOD
        
        # Calculate E_x = (1/N) * S_x
        E[x] = (current_sum * N_inv) % MOD
    
    # The final answer is E_0.
    sys.stdout.write(str(E[0]) + "
")

# Call the solve function to execute the program.
solve()