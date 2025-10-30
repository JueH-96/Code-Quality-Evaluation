MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    B = list(map(int, input[ptr:ptr+N]))
    
    q = B.count(-1)
    
    dp_prev = [0] * (M + 2)  # m ranges from 1 to M+1
    dp_prev[M+1] = 1  # initial state: min is M+1
    
    sum_components = 0
    inv_M = pow(M, MOD-2, MOD) if M != 0 else 0
    
    for num in B:
        new_dp = [0] * (M + 2)
        contribution = 0
        
        if num == -1:
            # Calculate sum_terms
            sum_terms = 0
            for m_prev in range(1, M+1):
                sum_terms = (sum_terms + (m_prev - 1) * dp_prev[m_prev]) % MOD
            sum_terms = (sum_terms + M * dp_prev[M+1]) % MOD
            contribution = sum_terms * inv_M % MOD
            
            # Compute suffix_sum
            suffix_sum = [0] * (M + 2)
            suffix_sum[M+1] = dp_prev[M+1]
            for m_prev in range(M, 0, -1):
                suffix_sum[m_prev] = (suffix_sum[m_prev + 1] + dp_prev[m_prev]) % MOD
            
            # Update new_dp
            for m_new in range(1, M+1):
                sum_prev_greater = suffix_sum[m_new + 1]
                sum_prev_equal = dp_prev[m_new]
                numerator = (sum_prev_greater + sum_prev_equal * (M - m_new + 1)) % MOD
                new_dp_val = numerator * inv_M % MOD
                new_dp[m_new] = new_dp_val
        else:
            val = num
            for m_prev in range(1, M+2):
                if dp_prev[m_prev] == 0:
                    continue
                new_min = min(m_prev, val)
                new_dp[new_min] = (new_dp[new_min] + dp_prev[m_prev]) % MOD
                if val < m_prev:
                    contribution = (contribution + dp_prev[m_prev]) % MOD
        
        sum_components = (sum_components + contribution) % MOD
        dp_prev = new_dp
    
    ans = sum_components * pow(M, q, MOD) % MOD
    print(ans)

if __name__ == '__main__':
    main()