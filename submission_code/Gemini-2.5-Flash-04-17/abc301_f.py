import sys

def solve():
    S = sys.stdin.readline().strip()
    MOD = 998244353
    N = len(S)

    # dp[j] = number of ways to fill the first i characters S[0...i-1]
    # such that the resulting string has the highest state j.
    # States j represent the highest type of subsequence matched:
    # 0: No UU subsequence
    # 1: UU subsequence exists, but no UUl subsequence
    # 2: UUl subsequence exists, but no UUlY subsequence
    # 3: UUlY subsequence exists (Forbidden state)

    # We want the sum of counts for states 0, 1, and 2 after processing the entire string.

    # dp state is [count_state_0, count_state_1, count_state_2, count_state_3]
    dp = [0] * 4
    # Base case: empty string (processing 0 characters)
    # The empty string has no UU subsequence, so it's in state 0. There is 1 way to have an empty string.
    dp[0] = 1 

    for i in range(N):
        char = S[i]
        next_dp = [0] * 4

        # Consider current state k and ways to reach it (dp[k])
        for k in range(4):
            ways = dp[k]
            if ways == 0:
                continue

            if 'a' <= char <= 'z': # Current character is lowercase
                # Transition based on current state k and adding a lowercase character
                if k == 0: # Was state 0 (no UU)
                    # Adding LC doesn't create UU. Stays state 0.
                    next_dp[0] = (next_dp[0] + ways) % MOD
                elif k == 1: # Was state 1 (UU, no UUl)
                    # Adding LC allows forming UUl subsequence. Transitions to state 2.
                    next_dp[2] = (next_dp[2] + ways) % MOD
                elif k == 2: # Was state 2 (UUl, no UUlY)
                    # Adding LC still results in UUl being the highest. Stays state 2.
                    next_dp[2] = (next_dp[2] + ways) % MOD
                elif k == 3: # Was state 3 (UUlY)
                    # Already in forbidden state. Stays state 3.
                    next_dp[3] = (next_dp[3] + ways) % MOD

            elif 'A' <= char <= 'Z': # Current character is uppercase
                # Transition based on current state k and adding an uppercase character
                if k == 0: # Was state 0 (no UU)
                    # Adding UC allows forming UU subsequence. Transitions to state 1.
                    next_dp[1] = (next_dp[1] + ways) % MOD
                elif k == 1: # Was state 1 (UU, no UUl)
                    # Adding UC still results in UU being the highest. Stays state 1.
                    next_dp[1] = (next_dp[1] + ways) % MOD
                elif k == 2: # Was state 2 (UUl, no UUlY)
                    # Adding UC allows forming UUlY subsequence. Transitions to state 3.
                    next_dp[3] = (next_dp[3] + ways) % MOD
                elif k == 3: # Was state 3 (UUlY)
                    # Already in forbidden state. Stays state 3.
                    next_dp[3] = (next_dp[3] + ways) % MOD

            else: # Current character is '?'
                # Case 1: '?' becomes lowercase (26 ways)
                # Transition based on current state k and adding a lowercase character
                if k == 0: # Was state 0 (no UU)
                    next_dp[0] = (next_dp[0] + ways * 26) % MOD
                elif k == 1: # Was state 1 (UU, no UUl)
                    next_dp[2] = (next_dp[2] + ways * 26) % MOD
                elif k == 2: # Was state 2 (UUl, no UUlY)
                    next_dp[2] = (next_dp[2] + ways * 26) % MOD
                elif k == 3: # Was state 3 (UUlY)
                    next_dp[3] = (next_dp[3] + ways * 26) % MOD

                # Case 2: '?' becomes uppercase (26 ways)
                # Transition based on current state k and adding an uppercase character
                if k == 0: # Was state 0 (no UU)
                    next_dp[1] = (next_dp[1] + ways * 26) % MOD
                elif k == 1: # Was state 1 (UU, no UUl)
                    next_dp[1] = (next_dp[1] + ways * 26) % MOD
                elif k == 2: # Was state 2 (UUl, no UUlY)
                    next_dp[3] = (next_dp[3] + ways * 26) % MOD
                elif k == 3: # Was state 3 (UUlY)
                    next_dp[3] = (next_dp[3] + ways * 26) % MOD
        
        dp = next_dp # Update dp for the next iteration

    # The number of strings that do NOT contain a DDoS-type string as a subsequence
    # is the sum of counts for states 0, 1, and 2 after processing the entire string.
    ans = (dp[0] + dp[1] + dp[2]) % MOD
    print(ans)

solve()