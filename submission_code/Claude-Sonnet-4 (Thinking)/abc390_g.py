def solve():
    N = int(input())
    MOD = 998244353
    
    # Compute number of digits for each number
    digits = [0] * (N + 1)
    for k in range(1, N + 1):
        digits[k] = len(str(k))
    
    # Group numbers by digit count
    count = {}
    sum_k = {}
    for k in range(1, N + 1):
        d = digits[k]
        if d not in count:
            count[d] = 0
            sum_k[d] = 0
        count[d] += 1
        sum_k[d] = (sum_k[d] + k) % MOD
    
    # Compute h_all[m] using DP
    # h_all[m] = coefficient of x^m in ∏(k=1 to N)(1 + x * 10^digits[k])
    h_all = [0] * (N + 1)
    h_all[0] = 1
    
    for k in range(1, N + 1):
        new_h_all = [0] * (N + 1)
        for m in range(N + 1):
            new_h_all[m] = h_all[m]
            if m > 0:
                new_h_all[m] = (new_h_all[m] + h_all[m - 1] * pow(10, digits[k], MOD)) % MOD
        h_all = new_h_all
    
    # Compute factorials
    factorial = [1] * N
    for i in range(1, N):
        factorial[i] = (factorial[i - 1] * i) % MOD
    
    # Compute d[j] = ∑(l=0 to N-1-j) (l+j)! * h_all[l]
    d = [0] * N
    for j in range(N):
        for l in range(N - j):
            if l + j < N:
                d[j] = (d[j] + factorial[l + j] * h_all[l]) % MOD
    
    # Compute the final sum
    total_sum = 0
    for j in range(N):
        sum_term = 0
        for digit_count in count:
            sum_term = (sum_term + sum_k[digit_count] * pow(10, j * digit_count, MOD)) % MOD
        
        sign = 1 if j % 2 == 0 else -1
        total_sum = (total_sum + sign * d[j] * sum_term) % MOD
    
    total_sum = (total_sum * factorial[N - 1]) % MOD
    return total_sum % MOD

print(solve())