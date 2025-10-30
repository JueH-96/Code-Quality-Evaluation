def solve():
    n = int(input())
    a = list(map(int, input().split()))
    mod = 998244353
    
    dp = {}
    dp[0] = 1
    
    for i in range(n):
        next_dp = {}
        for mask in dp:
            for outcome in range(1, a[i] + 1):
                sum_10_achieved = False
                if outcome == 10:
                    sum_10_achieved = True
                elif 1 <= 10 - outcome <= 9:
                    if (mask >> (10 - outcome - 1)) & 1:
                        sum_10_achieved = True
                        
                if not sum_10_achieved:
                    next_mask = mask
                    if 1 <= outcome <= 9:
                        next_mask |= (1 << (outcome - 1))
                    for j in range(1, 10):
                        if (mask >> (j - 1)) & 1:
                            if j + outcome <= 9:
                                next_mask |= (1 << (j + outcome - 1))
                    next_dp[next_mask] = next_dp.get(next_mask, 0) + dp[mask]
                    
        for mask in next_dp:
            next_dp[mask] %= mod
        dp = next_dp
        
    unfavorable_count = 0
    for mask in dp:
        unfavorable_count += dp[mask]
        unfavorable_count %= mod
        
    total_outcomes = 1
    for val in a:
        total_outcomes = (total_outcomes * val) % mod
        
    if total_outcomes == 0:
        if unfavorable_count == 0:
            print(0)
        else:
            print(0) # Should not happen based on problem constraints, but just in case.
        return
        
    total_probability_unfavorable = (unfavorable_count * pow(total_outcomes, mod - 2, mod)) % mod
    favorable_probability = (1 - total_probability_unfavorable + mod) % mod
    print(favorable_probability)

if __name__ == '__main__':
    solve()