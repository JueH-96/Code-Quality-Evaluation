# YOUR CODE HERE
import sys

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    # Read X as a 0-indexed list
    X = list(map(int, sys.stdin.readline().split())) 
    MOD = 998244353

    # Precompute factorials and inverse factorials for combinations `nCr_mod`
    # These are needed only for the special case M=2, X[0]==X[1]
    MAX_N_COMB = N # Max value needed for nCr arguments
    fact = [1] * (MAX_N_COMB + 1)
    inv = [1] * (MAX_N_COMB + 1) # modular inverse for i
    fact_inv = [1] * (MAX_N_COMB + 1)
    for i in range(1, MAX_N_COMB + 1):
        fact[i] = (fact[i-1] * i) % MOD
    
    # Compute modular inverse using Fermat's Little Theorem pow(a, MOD-2, MOD)
    # Calculate inverse factorial for N first
    fact_inv[MAX_N_COMB] = pow(fact[MAX_N_COMB], MOD - 2, MOD)
    # Calculate inverse factorials iteratively downwards
    for i in range(MAX_N_COMB - 1, -1, -1):
        fact_inv[i] = (fact_inv[i+1] * (i+1)) % MOD

    # Calculate modular inverses using inverse factorials
    # Not strictly needed for nCr, but can be useful
    # for i in range(1, MAX_N_COMB + 1): 
    #      inv[i] = (fact_inv[i] * fact[i-1]) % MOD
         
    def nCr_mod(n, r):
        """ Computes nCr modulo MOD """
        if r < 0 or r > n: return 0
        if r == 0 or r == n: return 1
        # Optimization: C(n, k) = C(n, n-k)
        if r > n // 2: r = n - r 
        
        # Using precomputed factorials and inverse factorials
        num = fact[n]
        den = (fact_inv[r] * fact_inv[n-r]) % MOD
        return (num * den) % MOD

    # Memoization for calculate_f function
    memo_f = {}
    def calculate_f(L, k):
        """ 
        Computes the number of sequences of length L using alphabet {1..k} 
        that contain all k distinct characters.
        Uses the Principle of Inclusion-Exclusion.
        """
        # Base cases
        if k < 0: return 0 # Invalid alphabet size
        if k == 0: return 1 if L == 0 else 0 # Empty alphabet: only empty sequence works
        if L < k : return 0 # Sequence length L too short to contain k distinct chars
         
        state = (L, k)
        if state in memo_f: return memo_f[state]
         
        res = 0
        # Apply Inclusion-Exclusion principle
        for i in range(k + 1):
             # Number of sequences using a subset of size (k-i) characters
             term = (nCr_mod(k, i) * pow(k - i, L, MOD)) % MOD
             if i % 2 == 1: # Subtract sets with odd number of missing characters
                 res = (res - term + MOD) % MOD
             else: # Add sets with even number of missing characters
                 res = (res + term) % MOD
        
        memo_f[state] = res
        return res


    def compute_dp_m_minus_1(target_seq):
        """ 
        Computes dp[N][M-1] for the given target_seq using matrix exponentiation.
        dp[N][j] = Number of sequences of length N whose longest prefix subsequence
                   match with target_seq has length j.
        Returns dp[N][M-1] where M is the length of target_seq.
        Uses a standard subsequence automaton construction (similar to KMP).
        """
        
        current_M = len(target_seq) # M for the specific sequence
        if current_M == 0: return 0 # Empty target sequence. dp[N][-1] is undefined/0.
        
        # Base case: If M=1, we need dp[N][0]. This is the count of sequences
        # of length N that do *not* contain target_seq[0].
        # Each position can be any of K-1 characters.
        if current_M == 1:
             return pow(K-1, N, MOD)

        # Build the subsequence automaton transition table `go[j][c]` and failure links `link[j]`
        go = [[0] * (K + 1) for _ in range(current_M + 1)]
        link = [0] * (current_M + 1)
        
        # Queue for BFS state processing
        q = [0] 
        head = 0
        processed_states = {0} # Track visited states to avoid cycles in BFS? Not needed with DAG structure?

        # Build automaton states level by level
        while head < len(q):
             j = q[head] # Current state (length of matched prefix)
             head += 1
             
             current_char_needed = -1 # Character needed to extend match from state j
             if j < current_M:
                 current_char_needed = target_seq[j] # 0-based index

             # Compute transitions for each character c
             for c in range(1, K + 1):
                 if j < current_M and c == current_char_needed:
                      # Match transition: move to state j+1
                      next_state = j + 1
                      go[j][c] = next_state
                      
                      # Update failure link for the new state `next_state` using KMP logic
                      if j == 0: # State 1's failure link is always state 0
                         link[next_state] = 0 
                      else:
                         # Follow failure link of current state j, then transition with character c
                         link[next_state] = go[link[j]][c]

                      # Add new state to queue if not processed
                      if next_state <= current_M and next_state not in processed_states:
                         q.append(next_state)
                         processed_states.add(next_state)
                 else:
                      # Mismatch transition: follow failure link and transition from there
                      go[j][c] = go[link[j]][c]
        
        # State M (or current_M) represents having found the full subsequence. It's an absorbing state.
        final_state_idx = current_M
        for c in range(1, K + 1):
             go[final_state_idx][c] = final_state_idx
        
        # Construct transition matrix T. T[k][j] = number of chars `c` causing transition j -> k
        T = [[0] * (current_M + 1) for _ in range(current_M + 1)]
        for j in range(current_M + 1):
            # Count transitions OUT of state j for each character
            counts = {} # Store counts for each destination state k
            for c in range(1, K + 1):
                k = go[j][c] # Destination state from j with char c
                counts[k] = counts.get(k, 0) + 1
            
            # Fill the matrix T column j based on counts
            for k, count in counts.items():
                 if 0 <= k <= current_M: # Ensure k is a valid state index
                     T[k][j] = count % MOD 

        # Matrix exponentiation to compute T^N
        def mat_mul(A, B, size):
            """ Multiplies two square matrices A and B of size `size` """
            C = [[0] * size for _ in range(size)]
            for i in range(size):
                for j in range(size):
                    sum_val = 0
                    # Matrix multiplication computation
                    A_row_i = A[i]
                    for l in range(size):
                        sum_val = (sum_val + A_row_i[l] * B[l][j]) % MOD
                    C[i][j] = sum_val
            return C

        def mat_pow(A, n, size):
            """ Computes A^n for square matrix A of size `size` using binary exponentiation """
            res = [[0] * size for _ in range(size)]
            # Initialize result matrix as identity matrix
            for i in range(size):
                res[i][i] = 1
            
            base = A
            while n > 0:
                if n % 2 == 1: # If n is odd
                    res = mat_mul(res, base, size)
                # Square the base matrix
                base = mat_mul(base, base, size)
                n //= 2 # Halve n
            return res

        # Handle N=0 case
        if N == 0:
             # If N=0, the only sequence is empty. dp[0][0]=1.
             # Need dp[0][M-1]. If M=1, target M-1=0, result 1. If M>1, target M-1 > 0, result 0.
             return 1 if current_M == 1 else 0 

        # Compute T^N
        T_N = mat_pow(T, N, current_M + 1)
        
        # The final DP state vector dp[N] is T_N * dp[0]. Since dp[0] = [1, 0, ..., 0]^T,
        # dp[N] is just the first column of T_N. dp[N][j] = T_N[j][0].
        
        # We need dp[N][M-1]. This corresponds to T_N[current_M-1][0].
        dp_N_M_minus_1 = T_N[current_M-1][0]
        
        return dp_N_M_minus_1


    # Main logic based on problem analysis and observed pattern from samples
    
    # Handle the special combinatorial case: M=2 and X[0] == X[1]
    if M == 2 and X[0] == X[1]:
        ans = 0
        # Required alphabet for prefix/suffix has size K-1
        target_k = K - 1 
        
        # K=1 implies target_k=0. The combinatorial logic handles this.
        # If K=1, X must be (1,1). Only possible char is 1. Any sequence A=(1,...,1).
        # If N>=2, A contains (1,1) as subsequence. So X is always a subsequence.
        # Condition "X not subsequence" is never met. Count is 0.
        if target_k < 0: # This implies K=0, impossible.
             pass
        else:
             # Sum over possible positions p for the single required character X[0]
             for p in range(1, N + 1): 
                 len_pref = p - 1
                 len_suff = N - p
                 
                 # Both prefix and suffix must contain all `target_k` characters
                 # Check if lengths are sufficient first
                 if len_pref < target_k or len_suff < target_k:
                      continue # Impossible to contain all chars

                 # Compute count for prefix and suffix using calculate_f
                 f_pref = calculate_f(len_pref, target_k)
                 f_suff = calculate_f(len_suff, target_k)
                 
                 # Combine counts
                 term = (f_pref * f_suff) % MOD
                 ans = (ans + term) % MOD
        
        print(ans)
        
    else: 
        # Apply the general rule derived from analysis of Samples 2, 3, 4
        # Case M >= 3 OR (M=2 and X[0] != X[1])
        
        count_X = compute_dp_m_minus_1(X)

        # The rule depends on comparing the last two elements of X
        # Check if M >= 2 before accessing X[M-2]
        if M >= 2 and X[M-1] != X[M-2]:
            # Construct sequence Y by replacing the last element of X with X[M-2]
            Y = X[:M-1] + [X[M-2]]
            count_Y = compute_dp_m_minus_1(Y)
            # Result is the difference modulo MOD
            result = (count_X - count_Y + MOD) % MOD
        else: 
            # Case M < 2 impossible by constraints (M>=2).
            # So this condition implies X[M-1] == X[M-2]
            result = count_X
        
        print(result)

solve()