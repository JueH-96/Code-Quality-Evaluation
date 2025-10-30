# YOUR CODE HERE
import sys

def solve():
    # Read inputs N and M
    N, M = map(int, sys.stdin.readline().split())
    # Define the modulus
    MOD = 998244353

    # === Step 1: Find all prime numbers <= M ===
    primes = []
    is_prime = [True] * (M + 1)
    # Handle edge cases M=0 and M=1 for is_prime array indexing
    if M >= 0: 
      if len(is_prime) > 0: is_prime[0] = False # 0 is not prime
      if len(is_prime) > 1: is_prime[1] = False # 1 is not prime
      
    # Sieve of Eratosthenes to find primes up to M
    for p in range(2, M + 1):
        if is_prime[p]:
            primes.append(p)
            # Mark multiples of p as not prime. Start from p*p for optimization.
            for i in range(p * p, M + 1, p):
                is_prime[i] = False
    
    num_primes = len(primes)
    
    # === Step 2: Handle edge case M=1 ===
    # If M=1, the only possible element in sequences is 1.
    # Any sequence (1, 1, ..., 1) has product 1. The score d(1) = 1.
    # There are sequences of length k=1 to N. Total N sequences.
    # The sum of scores is N * 1 = N.
    if M == 1:
         print(N % MOD)
         return

    # Map each prime to its index for easier access
    prime_map = {p: i for i, p in enumerate(primes)}

    # === Step 3: Determine state space based on LCM exponents ===
    # The state will be defined by the exponents of primes in the product,
    # but capped at the maximum exponent appearing in LCM(1, ..., M).
    # Compute maximum exponent for each prime p in LCM(1..M). This is floor(log_p M).
    lcm_exponents = [0] * num_primes
    for p_idx, p in enumerate(primes):
        power_val = p
        count = 0
        while power_val <= M:
             count +=1
             # Check to prevent potential overflow if M were large. Safe for M <= 16.
             if power_val > M // p : 
                 break
             power_val *= p
        lcm_exponents[p_idx] = count # max exponent for prime p

    # Calculate the size of the state space. A state is a tuple of exponents (e1, ..., er)
    # where 0 <= ei <= lcm_exponents[i]. The number of possible values for ei is lcm_exponents[i] + 1.
    state_dims = [exp + 1 for exp in lcm_exponents]
    state_size = 1
    for dim in state_dims:
        state_size *= dim

    # === Step 4: Create mappings for states ===
    # Map state tuples (exponent vectors) to unique integer indices and vice versa.
    state_to_idx = {}
    idx_to_state = [()] * state_size # Preallocate list
    
    current_idx = 0
    
    # Recursive function to generate all possible state tuples and assign indices
    def generate_states(p_idx, current_state_tuple):
        nonlocal current_idx
        # Base case: If all primes considered, store the state tuple and its index
        if p_idx == num_primes:
            state_to_idx[current_state_tuple] = current_idx
            idx_to_state[current_idx] = current_state_tuple
            current_idx += 1
            return

        # Recursive step: Iterate through all possible exponent values for the current prime
        for exp_val in range(state_dims[p_idx]): # Exponent ranges from 0 to lcm_exponents[p_idx]
             generate_states(p_idx + 1, current_state_tuple + (exp_val,))

    # Start state generation
    generate_states(0, tuple())

    # === Step 5: Precompute exponent vectors for numbers 1 to M ===
    # For each x in {1, ..., M}, find its prime factorization exponent vector.
    val_exponents = []
    for x in range(1, M + 1):
        exps = [0] * num_primes
        temp_x = x
        for p_idx, p in enumerate(primes):
            count = 0
            # Count the exponent of prime p in the factorization of x
            while temp_x > 0 and temp_x % p == 0:
                count += 1
                temp_x //= p
            exps[p_idx] = count
        val_exponents.append(tuple(exps))

    # === Step 6: Build the transition matrix T ===
    # T[i][j] = number of elements x in {1..M} that transition state i to state j.
    T = [[0] * state_size for _ in range(state_size)]

    for curr_idx in range(state_size):
        curr_state_tuple = idx_to_state[curr_idx] # Current state exponent tuple
        
        # Iterate through each possible next element x in the sequence
        for x in range(1, M + 1):
            x_exps = val_exponents[x-1] # Exponents of x
            next_state_list = list(curr_state_tuple) # Start with current exponents
            
            # Calculate the next state's exponents by adding x's exponents
            for p_idx in range(num_primes):
                 next_state_list[p_idx] += x_exps[p_idx]
                 # Cap the exponent at the maximum value from LCM(1..M)
                 next_state_list[p_idx] = min(next_state_list[p_idx], lcm_exponents[p_idx])
            
            next_state_tuple = tuple(next_state_list)
            # Find the index corresponding to the computed next state tuple
            next_idx = state_to_idx[next_state_tuple] 
            
            # Increment the transition count from curr_idx to next_idx
            T[curr_idx][next_idx] = (T[curr_idx][next_idx] + 1) % MOD

    # === Step 7: Matrix Exponentiation Utilities ===
    # Matrix multiplication function
    def mat_mul(A, B, size):
        C = [[0] * size for _ in range(size)]
        for i in range(size):
             A_row_i = A[i] # Cache row A[i] for minor optimization
             for j in range(size):
                 sum_val = 0
                 for k in range(size):
                    sum_val = (sum_val + A_row_i[k] * B[k][j]) % MOD
                 C[i][j] = sum_val
        return C

    # Matrix power function using binary exponentiation (exponentiation by squaring)
    def mat_pow(A, n, size):
        res = [[0] * size for _ in range(size)]
        for i in range(size):
            res[i][i] = 1 # Initialize result as Identity matrix
        
        base = A
        while n > 0:
            if n % 2 == 1: # If current exponent bit is 1
                res = mat_mul(res, base, size)
            base = mat_mul(base, base, size) # Square the base matrix
            n //= 2
        return res

    # === Step 8: Compute Sum of Matrix Powers ===
    # Use the augmented matrix trick to compute Sum_{k=1..N} T^k efficiently.
    # The augmented matrix is M_1 = [[T, T], [0, I]] of size (2*state_size) x (2*state_size).
    # Then (M_1)^N = [[T^N, Sum_{k=1..N} T^k], [0, I]].
    
    aug_state_size = 2 * state_size
    Aug_T = [[0] * aug_state_size for _ in range(aug_state_size)]
    
    # Populate the augmented matrix M_1
    for r in range(state_size):
        T_row_r = T[r] # Cache row T[r]
        for c in range(state_size):
             Aug_T[r][c] = T_row_r[c]             # Top-left block is T
             Aug_T[r][c + state_size] = T_row_r[c] # Top-right block is T
    
    for r in range(state_size):
         # Bottom-left block is 0 (already initialized)
         Aug_T[r + state_size][r + state_size] = 1 # Bottom-right block is I (Identity)

    # Handle N=0 case (though problem constraints N>=1)
    if N == 0:
        print(0)
        return

    # Compute (M_1)^N using matrix power function
    Aug_N = mat_pow(Aug_T, N, aug_state_size)
    
    # Extract the Sum_T = Sum_{k=1..N} T^k from the top-right block of Aug_N
    Sum_T = [[0] * state_size for _ in range(state_size)]
    for r in range(state_size):
        for c in range(state_size):
             Sum_T[r][c] = Aug_N[r][c + state_size]

    # === Step 9: Calculate the final result ===
    # The initial state corresponds to an empty sequence (product 1), which has exponent vector (0,0,...,0).
    initial_state_tuple = tuple([0]*num_primes)
    # Need to handle the case where M might be such that there are no primes (e.g., M=1, already handled).
    # If M>=2, there is at least one prime and the state (0,...,0) exists.
    if initial_state_tuple not in state_to_idx:
         # This should not be reached for M >= 2. Put error for robustness.
         print("Error: Initial state (0,...,0) not found in state map.") 
         return
    initial_state_idx = state_to_idx[initial_state_tuple]
    
    # The vector `final_counts` stores the total number of times each state `i` is reached,
    # summed over all sequence lengths k=1 to N, starting from the initial state.
    # This is obtained from the `initial_state_idx`-th column of `Sum_T`.
    # final_counts[i] = (Sum_T * V_0)_i where V_0 is basis vector for initial state.
    final_counts = [Sum_T[i][initial_state_idx] for i in range(state_size)]

    # Compute the total score sum. For each state, multiply the total count by the score associated with that state.
    total_score = 0
    for idx in range(state_size):
        state_tuple = idx_to_state[idx] # Get the exponent tuple for state `idx`
        
        # The score associated with a state (capped exponent tuple) is product of (exponent + 1).
        # This relies on the property that this method correctly computes the sum of scores.
        current_score = 1
        for exp_val in state_tuple:
             current_score = (current_score * (exp_val + 1)) % MOD
        
        # Add contribution: (number of times state `idx` is reached) * (score of state `idx`)
        total_score = (total_score + final_counts[idx] * current_score) % MOD
        
    # Print the final total score modulo MOD
    print(total_score)

# Execute the solve function
solve()