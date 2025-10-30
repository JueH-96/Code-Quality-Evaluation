from functools import lru_cache
from collections import defaultdict

MOD = 10**9 + 7

def compute_steps(c):
    if c == 1:
        return 0
    cnt = bin(c).count('1')
    return 1 + compute_steps(cnt)

def main():
    s = input().strip()
    n = len(s)
    bits = [int(c) for c in s]
    
    # Precompute compute_steps for all c up to 800
    max_c = 800
    steps_c = [0] * (max_c + 1)
    for c in range(1, max_c + 1):
        steps_c[c] = compute_steps(c)
    
    # Now, steps for x is 1 + steps_c[c]
    
    # Initialize DP
    dp = [{} for _ in range(n + 1)]
    dp[0][(True, True, 0)] = 1
    
    for pos in range(n):
        current_pos = pos
        for state in list(dp[pos].keys()):
            tight, leading_zero, c = state
            count = dp[pos][state]
            
            max_bit = bits[pos] if tight else 1
            for bit in [0, 1]:
                if bit > max_bit:
                    continue
                new_tight = tight and (bit == max_bit)
                new_leading_zero = leading_zero and (bit == 0)
                if new_leading_zero:
                    new_c = c
                else:
                    new_c = c + (bit == 1)
                    if new_c > max_c:
                        continue  # Since we precomputed up to 800, no need to check
                next_pos = current_pos + 1
                next_state = (new_tight, new_leading_zero, new_c)
                dp[next_pos][next_state] = (dp[next_pos].get(next_state, 0) + count) % MOD
    
    # Sum the counts for each c where 1 + steps_c[c] <=k
    total = 0
    for c in range(1, max_c + 1):
        if 1 + steps_c[c] <= k:
            total = (total + dp[n].get((True, True, c), 0)) % MOD
    
    print(total % MOD)

if __name__ == "__main__":
    k = int(input())
    main()