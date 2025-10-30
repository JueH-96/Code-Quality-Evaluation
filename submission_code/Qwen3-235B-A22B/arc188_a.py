import sys
from collections import defaultdict

MOD = 998244353

def is_good(a, b, c):
    par_a = a % 2
    par_b = b % 2
    par_c = c % 2
    if not (par_a == par_b == par_c):
        return False
    par = par_a
    if par == 1:
        m = min(a, b, c)
        if m < 1:
            return False
    return True

def main():
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    
    dp = defaultdict(int)
    # Initial state: count 0, no suffixes
    key = (0, tuple())
    dp[key] = 1
    
    for pos in range(N):
        current_char = S[pos]
        dp_new = defaultdict(int)
        # Process each existing state
        for state in dp:
            prev_count, suffix_tuple = state
            ways = dp[state]
            suffix_dict = dict(suffix_tuple)
            
            # Determine possible substitutions
            if current_char == '?':
                subs = ['A', 'B', 'C']
            else:
                subs = [current_char]
            
            for c in subs:
                new_suffixes = defaultdict(int)
                # Add new single-character suffix
                a, b, cc = 0, 0, 0
                if c == 'A':
                    a = 1
                elif c == 'B':
                    b = 1
                else:
                    cc = 1
                new_suffixes[(a, b, cc)] += 1
                
                # Extend previous suffixes
                for (a_s, b_s, c_s), freq in suffix_dict.items():
                    new_a = a_s + (1 if c == 'A' else 0)
                    new_b = b_s + (1 if c == 'B' else 0)
                    new_c = c_s + (1 if c == 'C' else 0)
                    new_suffixes[(new_a, new_b, new_c)] += freq
                
                # Calculate new_contrib
                new_contrib = 0
                for (a_s, b_s, c_s) in new_suffixes:
                    if is_good(a_s, b_s, c_s):
                        new_contrib += new_suffixes[(a_s, b_s, c_s)]
                
                # Update new_capped_count
                new_total = prev_count + new_contrib
                new_capped = K if new_total >= K else new_total
                
                # Convert new_suffixes to sorted tuple
                new_suffix_list = sorted(new_suffixes.items())
                new_key = (new_capped, tuple(new_suffix_list))
                dp_new[new_key] = (dp_new[new_key] + ways) % MOD
        
        dp = dp_new
    
    # Sum all states where count >= K (which are stored as K)
    ans = 0
    for (count, _), val in dp.items():
        if count >= K:
            ans = (ans + val) % MOD
    print(ans)

if __name__ == "__main__":
    main()