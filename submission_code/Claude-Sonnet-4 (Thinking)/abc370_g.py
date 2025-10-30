def solve():
    N, M = map(int, input().split())
    MOD = 998244353
    
    # Check if a number is good (sum of divisors divisible by 3)
    def is_good(n):
        divisor_sum = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                divisor_sum += i
                if i != n // i:
                    divisor_sum += n // i
            i += 1
        return divisor_sum % 3 == 0
    
    # Determine maximum k (elements > 1)
    max_k = 0
    temp = 1
    while temp * 2 <= N and max_k < M:
        temp *= 2
        max_k += 1
    max_k = min(max_k + 5, M)
    
    # Precompute binomial coefficients
    C = [[0] * (max_k + 1) for _ in range(M + 1)]
    for i in range(M + 1):
        C[i][0] = 1
        for j in range(1, min(i, max_k) + 1):
            C[i][j] = (C[i-1][j-1] + C[i-1][j]) % MOD
    
    # Count sequences with exactly k elements > 1
    def count_sequences_with_k_nonone_elements(k):
        if k == 0:
            return 1 if is_good(1) else 0
        
        # DP: dp[i][product] = ways to get product using i elements > 1
        dp = [{}]
        dp[0][1] = 1
        
        for i in range(k):
            dp.append({})
            for product in dp[i]:
                count = dp[i][product]
                max_element = N // product
                for next_element in range(2, max_element + 1):
                    new_product = product * next_element
                    if new_product <= N:
                        if new_product not in dp[i+1]:
                            dp[i+1][new_product] = 0
                        dp[i+1][new_product] = (dp[i+1][new_product] + count) % MOD
        
        result = 0
        for product in dp[k]:
            if is_good(product):
                result = (result + dp[k][product]) % MOD
        
        return result
    
    result = 0
    for k in range(max_k + 1):
        ways_for_k = count_sequences_with_k_nonone_elements(k)
        if k <= M:
            ways_to_choose_positions = C[M][k]
            result = (result + ways_for_k * ways_to_choose_positions) % MOD
    
    return result

print(solve())