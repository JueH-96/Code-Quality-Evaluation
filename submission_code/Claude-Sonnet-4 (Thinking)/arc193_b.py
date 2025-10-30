def solve():
    MOD = 998244353
    
    N = int(input())
    s = input().strip()
    
    k = s.count('1')
    
    # Compute f(N) = number of distinct cycle in-degree sequences
    f_N = count_realizable_sequences_fast(N)
    
    # Answer is f(N) * 2^k
    result = (f_N * pow(2, k, MOD)) % MOD
    print(result)

def count_realizable_sequences_fast(N):
    MOD = 998244353
    
    # DP1[i][s] = sequences of length i, sum s, where partial sums >= positions
    max_s1 = 2 * (N - 1)
    dp1_prev = [0] * (max_s1 + 1)
    if max_s1 >= 0:
        dp1_prev[0] = 1
    
    for i in range(1, N):
        dp1_curr = [0] * (max_s1 + 1)
        for s in range(i, max_s1 + 1):  # constraint: s >= i
            for c in [0, 1, 2]:
                if s - c >= 0 and s - c < len(dp1_prev):
                    dp1_curr[s] = (dp1_curr[s] + dp1_prev[s - c]) % MOD
        dp1_prev = dp1_curr
    
    # DP2[i][s] = sequences of length i, sum s, where partial sums <= positions  
    max_s2 = N - 1
    dp2_prev = [0] * (max_s2 + 1)
    if max_s2 >= 0:
        dp2_prev[0] = 1
    
    for i in range(1, N):
        dp2_curr = [0] * (max_s2 + 1)
        for s in range(min(i, max_s2) + 1):  # constraint: s <= i
            for c in [0, 1, 2]:
                if s - c >= 0:
                    dp2_curr[s] = (dp2_curr[s] + dp2_prev[s - c]) % MOD
        dp2_prev = dp2_curr
    
    count = 0
    
    # Try each possible value of c0
    for c0 in [0, 1, 2]:
        if N - c0 >= 0:
            s = N - c0
            # Overlap occurs when both conditions give same sequence (all partial sums = positions)
            # This happens when s = N-1 (each element = 1)
            overlap = 1 if s == N - 1 else 0
            
            val1 = dp1_prev[s] if s < len(dp1_prev) else 0
            val2 = dp2_prev[s] if s < len(dp2_prev) else 0
            
            count = (count + val1 + val2 - overlap) % MOD
    
    return count

solve()