# YOUR CODE HERE
import sys, math
from itertools import product

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N, K, P = map(int, sys.stdin.readline().split())
    plans = []
    for _ in range(N):
        parts = list(map(int, sys.stdin.readline().split()))
        C_i = parts[0]
        A_i = parts[1:]
        plans.append( (C_i, A_i) )
    
    base = P +1
    powers = [1]
    for _ in range(K-1):
        powers.append(powers[-1]*base)
    
    def encode(state):
        idx = 0
        for i in range(K):
            idx += state[i]*powers[i]
        return idx
    
    total_states = base**K
    INF = float('inf')
    dp = [INF]*total_states
    initial_state = tuple([0]*K)
    dp[encode(initial_state)] = 0
    
    for C_i, A_i in plans:
        prev_dp = dp.copy()
        for state_idx in range(total_states):
            if prev_dp[state_idx] < INF:
                # Decode state_idx to state
                state = []
                tmp = state_idx
                for _ in range(K):
                    state.append(tmp % base)
                    tmp = tmp // base
                state = state[:K]
                # Compute new state
                new_state = []
                for j in range(K):
                    new_val = state[j] + A_i[j]
                    if new_val > P:
                        new_val = P
                    new_state.append(new_val)
                new_state = tuple(new_state)
                new_idx = encode(new_state)
                if dp[new_idx] > prev_dp[state_idx] + C_i:
                    dp[new_idx] = prev_dp[state_idx] + C_i
    final_state = tuple([P]*K)
    final_idx = encode(final_state)
    if dp[final_idx] == INF:
        print(-1)
    else:
        print(dp[final_idx])

if __name__ == "__main__":
    main()