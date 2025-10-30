def solve():
    N = int(input())
    A = list(map(int, input().split()))
    digits = [len(str(a)) for a in A]  # Precompute the number of digits for each element
    
    MOD = 998244353
    # Precompute powers of 10 for efficiency (A_i can be at most 10^9, so at most 10 digits)
    powers_of_10 = [pow(10, i, MOD) for i in range(11)]
    
    total = 0
    
    for i in range(N-1):
        for j in range(i+1, N):
            # f(A_i, A_j) = A_i * 10^(digits of A_j) + A_j
            concatenated = (A[i] * powers_of_10[digits[j]] + A[j]) % MOD
            total = (total + concatenated) % MOD
    
    return total

print(solve())