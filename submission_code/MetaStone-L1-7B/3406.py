MOD = 10**9 + 7

def count_stable_arrays(zero, one, limit):
    n = zero + one
    if n == 0:
        return 1 if zero == 0 and one == 0 else 0
    if zero == 0 or one == 0:
        return 0
    
    dp = [{} for _ in range(n)]
    dp[0][(None, 0, 0, 0, 0)] = 1
    
    for i in range(n):
        for state in dp[i]:
            last_char, current_run, c0, c1 = state
            count = dp[i][state]
            
            for next_char in [0, 1]:
                if next_char == last_char:
                    new_run = current_run + 1
                    if new_run > limit:
                        continue
                else:
                    new_run = 1
                
                new_c0 = c0 + (next_char == 0)
                new_c1 = c1 + (next_char == 1)
                
                if new_c0 > zero or new_c1 > one:
                    continue
                
                new_state = (next_char, new_run, new_c0, new_c1)
                if new_state in dp[i+1]:
                    dp[i+1][new_state] = (dp[i+1][new_state] + count) % MOD
                else:
                    dp[i+1][new_state] = count % MOD
    
    total = 0
    for state in dp[n-1]:
        last_char, current_run, c0, c1 = state
        if c0 == zero and c1 == one:
            total = (total + dp[n-1][state]) % MOD
    
    return total

# Example usage:
# print(count_stable_arrays(1, 1, 2))  # Output: 2
# print(count_stable_arrays(1, 2, 1))  # Output: 1
# print(count_stable_arrays(3, 3, 2))  # Output: 14