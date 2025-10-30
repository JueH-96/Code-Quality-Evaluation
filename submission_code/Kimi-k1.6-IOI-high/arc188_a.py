import sys
from collections import defaultdict

def main():
    MOD = 998244353
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    
    # Pair index for each parity (0-7)
    pair_index = [0, 1, 2, 3, 3, 2, 1, 0]
    initial_pair = (1, 0, 0, 0)
    
    dp = defaultdict(int)
    dp[(0, initial_pair, 0)] = 1
    
    for c in S:
        new_dp = defaultdict(int)
        for (parity_p, pair_counts, accum), cnt in dp.items():
            # Determine possible characters
            if c == '?':
                chars = ['A', 'B', 'C']
            else:
                chars = [c]
            for char in chars:
                # Determine mask based on character
                if char == 'A':
                    mask = 1
                elif char == 'B':
                    mask = 2
                else:
                    mask = 4
                new_parity = parity_p ^ mask
                pi = pair_index[new_parity]
                contrib = pair_counts[pi]
                new_accum = accum + contrib
                if new_accum >= K:
                    new_accum = K
                # Update pair counts
                new_pair = list(pair_counts)
                new_pair[pi] += 1
                new_pair_tuple = tuple(new_pair)
                # Update new_dp
                key = (new_parity, new_pair_tuple, new_accum)
                new_dp[key] = (new_dp[key] + cnt) % MOD
        dp = new_dp
        if not dp:
            break  # No possible states
    
    result = 0
    for state, cnt in dp.items():
        _, _, accum = state
        if accum >= K:
            result = (result + cnt) % MOD
    print(result % MOD)

if __name__ == "__main__":
    main()