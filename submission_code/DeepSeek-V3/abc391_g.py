import sys
MOD = 998244353

def solve():
    N, M = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    
    # The DP state is a tuple representing the current LCS dp array for S.
    # For N up to 10, we can represent the state as a tuple of N+1 elements (0..N), but optimized.
    # The initial state is (0, 0, ..., 0) (or actually, the initial state is [0]*N)
    # But for LCS tracking, the state is the current dp array where dp[i] is the length of the LCS of the processed string and S[0..i-1].
    # However, for N=10, the state can be represented as a tuple of N elements (since S has length N).
    
    from collections import defaultdict
    
    # Initial state: before processing any characters, the LCS is 0 for all prefixes.
    # The state is a tuple of (l_0, l_1, ..., l_{N-1}), where l_i is the LCS length for S[0..i].
    # Initially, all l_i are 0.
    initial_state = tuple([0]*N)
    
    dp = defaultdict(int)
    dp[initial_state] = 1
    
    for _ in range(M):
        new_dp = defaultdict(int)
        for state in dp:
            current_count = dp[state]
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_state = list(state)
                for i in range(N):
                    if i == 0:
                        if S[i] == c:
                            new_state[i] = 1 if new_state[i] == 0 else max(new_state[i], 1)
                    else:
                        if S[i] == c:
                            new_state[i] = max(new_state[i], new_state[i-1] + 1)
                        new_state[i] = max(new_state[i], new_state[i-1])
                new_state_tuple = tuple(new_state)
                new_dp[new_state_tuple] = (new_dp[new_state_tuple] + current_count) % MOD
        dp = new_dp
    
    # Now, collect all states and their counts. For each state, the LCS length is state[-1].
    ans = [0]*(N+1)
    for state in dp:
        lcs_length = state[-1]
        ans[lcs_length] = (ans[lcs_length] + dp[state]) % MOD
    
    print(' '.join(map(str, ans)))

solve()