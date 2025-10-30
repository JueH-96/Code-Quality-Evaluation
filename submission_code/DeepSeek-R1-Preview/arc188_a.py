import sys
from collections import defaultdict

MOD = 998244353

def main():
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    
    # Initialize the previous parities and dp_prev
    prev_parities = defaultdict(int)
    prev_parities[(0, 0, 0)] = 1  # before any characters
    
    dp_prev = defaultdict(int)
    dp_prev[(0, 0, 0, 0)] = 1  # (a, b, c, total), count
    
    for j in range(N):
        current_char = S[j]
        choices = ['A', 'B', 'C'] if current_char == '?' else [current_char]
        
        current_dp = defaultdict(int)
        
        for choice in choices:
            for (a_prev, b_prev, c_prev, total_prev), ways in dp_prev.items():
                # Compute new_a, new_b, new_c
                new_a = (a_prev + (1 if choice == 'A' else 0)) % 2
                new_b = (b_prev + (1 if choice == 'B' else 0)) % 2
                new_c = (c_prev + (1 if choice == 'C' else 0)) % 2
                
                # Compute new_substrings by checking prev_parities
                new_substrings = 0
                for (a_pp, b_pp, c_pp), count in prev_parities.items():
                    a_diff = (new_a + a_pp) % 2
                    b_diff = (new_b + b_pp) % 2
                    c_diff = (new_c + c_pp) % 2
                    if a_diff == b_diff == c_diff:
                        new_substrings = (new_substrings + count) % MOD
                
                new_total = (total_prev + new_substrings) % MOD
                key = (new_a, new_b, new_c, new_total)
                current_dp[key] = (current_dp.get(key, 0) + ways) % MOD
        
        # Update prev_parities for the next step
        new_parities = defaultdict(int)
        for (a, b, c, total), ways in current_dp.items():
            par_key = (a, b, c)
            new_parities[par_key] = (new_parities[par_key] + ways) % MOD
        
        prev_parities = new_parities
        dp_prev = current_dp
    
    # Calculate the result
    result = 0
    for (a, b, c, total), ways in dp_prev.items():
        if total >= K:
            result = (result + ways) % MOD
    print(result)

if __name__ == '__main__':
    main()