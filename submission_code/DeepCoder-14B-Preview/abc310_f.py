MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    a_list = list(map(int, data[1:n+1]))
    
    # Calculate the total number of possible outcomes
    total = 1
    for a in a_list:
        total = (total * a) % MOD
    
    # Initialize DP: key is the state (bitmask), value is the count
    dp = {0: 1}  # Initially, no sums are achievable
    
    for a in a_list:
        new_dp = {}
        for state, cnt in dp.items():
            # Compute forbidden a_i's
            forbidden = set()
            forbidden.add(10)  # a_i cannot be 10
            # Add 10 - s for each sum s in the current state
            for s in range(1, 11):
                if (state >> s) & 1:
                    forbidden.add(10 - s)
            
            # Process a_i > 10
            cnt_greater = max(0, a - 10)
            if cnt_greater > 0:
                # The state remains the same
                if state in new_dp:
                    new_dp[state] = (new_dp[state] + cnt * cnt_greater) % MOD
                else:
                    new_dp[state] = (cnt * cnt_greater) % MOD
            
            # Process a_i <= min(a, 10)
            max_a_i = min(a, 10)
            for ai in range(1, max_a_i + 1):
                if ai in forbidden:
                    continue  # Skip forbidden a_i's
                
                # Compute the new state
                new_state = state  # Start with the current state
                # Add ai itself if <=10
                if ai <= 10:
                    new_state |= (1 << ai)
                # Add ai to all existing sums in the state
                for s in range(1, 11):
                    if (state >> s) & 1:
                        new_sum = s + ai
                        if new_sum <= 10:
                            new_state |= (1 << new_sum)
                
                # Update the new_dp
                if new_state in new_dp:
                    new_dp[new_state] = (new_dp[new_state] + cnt) % MOD
                else:
                    new_dp[new_state] = cnt % MOD
        
        dp = new_dp
    
    # Calculate the number of bad outcomes: no subset sums to 10
    bad = 0
    for state, cnt in dp.items():
        if (state & (1 << 10)) == 0:  # 10th bit represents sum 10
            bad = (bad + cnt) % MOD
    
    # Calculate the number of good outcomes
    good = (total - bad) % MOD
    
    # Calculate the result as (good / total) mod MOD
    if total == 0:
        print(0)
        return
    
    inv_total = pow(total, MOD - 2, MOD)
    ans = (good * inv_total) % MOD
    print(ans)

if __name__ == '__main__':
    main()