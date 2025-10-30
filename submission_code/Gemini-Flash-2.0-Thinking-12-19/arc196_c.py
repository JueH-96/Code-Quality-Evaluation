import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    MOD = 998244353

    d = [0] * (2 * N + 1)
    for i in range(2 * N):
        if S[i] == 'W':
            d[i+1] = d[i] + 1
        else:
            d[i+1] = d[i] - 1

    # Check if total W and B are N each
    if d[2 * N] != 0:
        print(0)
        return

    firstW = S.find('W') # 0-indexed
    lastW = S.rfind('W') # 0-indexed
    firstB = S.find('B') # 0-indexed
    lastB = S.rfind('B') # 0-indexed

    # dp[i % 2][j] stores the number of ways to process vertices 1...i resulting in
    # j white vertices <= i paired with black vertices > i.
    # Number of black vertices <= i paired with white vertices > i is j - d[i].
    dp = [[0] * (N + 1) for _ in range(2)]
    dp[0][0] = 1 # Base case: 0 vertices processed, 0 unmatched of either type

    for i in range(1, 2 * N + 1):
        curr = i % 2
        prev = (i - 1) % 2
        
        min_j_prime = max(0, d[i-1])
        max_j_prime_num = i - 1 + d[i-1]
        if max_j_prime_num % 2 != 0:
             max_j_prime = -1
        else:
             max_j_prime = max_j_prime_num // 2

        for j_prime in range(min_j_prime, max_j_prime + 1):
            if dp[prev][j_prime] == 0:
                continue

            u_b_prime = j_prime - d[i-1] # number of B <= i-1 paired with W > i-1
            
            if S[i-1] == 'W': # Vertex i is W
                # Option 1: Pair i (W<=i) with k<i (B<=i). k was B<=i-1 paired with W>i-1.
                # Number of choices for k is u_b_prime.
                # New state j = j_prime. New k = u_b_prime - 1. (k is num B<=i paired with W>i)
                next_j = j_prime
                current_k = u_b_prime - 1
                
                # Check range for next_j and current_k
                valid_transition = (u_b_prime >= 1) and (next_j >= max(0, d[i])) and (current_k >= max(0, -d[i]))

                if valid_transition:
                    sc_ok = True
                    if i > 0 and i < 2*N:
                        # SC condition for cut (1..i, i+1..2N)
                        # Need j >= 1 if W<=i non-empty and B>i non-empty
                        if (i >= firstW + 1) and (i <= lastB) and (next_j == 0): sc_ok = False
                        # Need k >= 1 if B<=i non-empty and W>i non-empty
                        if (i >= firstB + 1) and (i <= lastW) and (current_k == 0): sc_ok = False

                    if sc_ok:
                         dp[curr][next_j] = (dp[curr][next_j] + dp[prev][j_prime] * u_b_prime) % MOD

                # Option 2: Pair i (W<=i) with k>i (B>i). 1 way.
                # New state j = j_prime + 1. New k = u_b_prime. (k is num B<=i paired with W>i)
                next_j = j_prime + 1
                current_k = u_b_prime
                
                # Check range for next_j and current_k
                valid_transition = (next_j <= N) and (next_j >= max(0, d[i])) and (current_k >= max(0, -d[i]))

                if valid_transition:
                    sc_ok = True
                    if i > 0 and i < 2*N:
                        # SC condition for cut (1..i, i+1..2N)
                        if (i >= firstW + 1) and (i <= lastB) and (next_j == 0): sc_ok = False
                        if (i >= firstB + 1) and (i <= lastW) and (current_k == 0): sc_ok = False
                    
                    if sc_ok:
                        dp[curr][next_j] = (dp[curr][next_j] + dp[prev][j_prime]) % MOD

            else: # Vertex i is B
                # Option 1: Pair i (B<=i) with k<i (W<=i). k was W<=i-1 paired with B>i-1.
                # Number of choices for k is j_prime.
                # New state j = j_prime - 1. New k = u_b_prime. (k is num B<=i paired with W>i)
                next_j = j_prime - 1
                current_k = u_b_prime
                
                # Check range for next_j and current_k
                valid_transition = (j_prime >= 1) and (next_j >= max(0, d[i])) and (current_k >= max(0, -d[i]))

                if valid_transition:
                    sc_ok = True
                    if i > 0 and i < 2*N:
                        # SC condition for cut (1..i, i+1..2N)
                         if (i >= firstW + 1) and (i <= lastB) and (next_j == 0): sc_ok = False
                         if (i >= firstB + 1) and (i <= lastW) and (current_k == 0): sc_ok = False
                    
                    if sc_ok:
                         dp[curr][next_j] = (dp[curr][next_j] + dp[prev][j_prime] * j_prime) % MOD

                # Option 2: Pair i (B<=i) with k>i (W>i). 1 way.
                # New state j = j_prime. New k = u_b_prime + 1. (k is num B<=i paired with W>i)
                next_j = j_prime
                current_k = u_b_prime + 1
                
                # Check range for next_j and current_k
                valid_transition = (current_k <= N) and (next_j >= max(0, d[i])) and (current_k >= max(0, -d[i]))
                
                if valid_transition:
                    sc_ok = True
                    if i > 0 and i < 2*N:
                         if (i >= firstW + 1) and (i <= lastB) and (next_j == 0): sc_ok = False
                         if (i >= firstB + 1) and (i <= lastW) and (current_k == 0): sc_ok = False
                    
                    if sc_ok:
                        dp[curr][next_j] = (dp[curr][next_j] + dp[prev][j_prime]) % MOD

    print(dp[2*N % 2][0])

solve()