def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # We need to compute efficiently the sum of f(A_i, A_j) for all 1 <= i < j <= N
    # f(A_i, A_j) = int(str(A_i) + str(A_j))
    
    # To optimize, we need to avoid string operations in the innermost loop.
    # We can precompute the length of each number in decimal form.
    lengths = [len(str(a)) for a in A]
    
    # Precompute powers of 10 up to 10^10 (since max A_i is 10^9, max length is 10)
    power_of_10 = [1] * 11
    for i in range(1, 11):
        power_of_10[i] = (power_of_10[i-1] * 10) % MOD
    
    # We will use the formula:
    # f(A_i, A_j) = A_i * 10^len(A_j) + A_j
    # We need to sum this for all i < j
    
    result = 0
    
    # We can use a frequency count of lengths to optimize the multiplication
    # For each A_i, we want to know how many times we will multiply it by 10^len(A_j)
    # for each possible length of A_j.
    
    # Count how many numbers have each length
    count_by_length = [0] * 11
    for l in lengths:
        count_by_length[l] += 1
    
    # We will accumulate sums by length to avoid recomputing sums multiple times
    sum_by_length = [0] * 11
    for i in range(N):
        l = lengths[i]
        sum_by_length[l] = (sum_by_length[l] + A[i]) % MOD
    
    # Now compute the result
    for i in range(N):
        Ai = A[i]
        lenAi = lengths[i]
        
        # For each length, compute contribution of Ai with all Aj where j > i
        for lenB in range(1, 11):
            # Ai contributes to all pairs (Ai, Aj) where len(Aj) = lenB
            # Contribution is Ai * 10^lenB
            Ai_contrib = Ai * power_of_10[lenB] % MOD
            # We add this contribution for each Aj of length lenB
            # But we need to exclude the current Ai if its length is lenB and it's counted in sum_by_length[lenB]
            if lenAi == lenB:
                result = (result + Ai_contrib * ((sum_by_length[lenB] - Ai) % MOD)) % MOD
            else:
                result = (result + Ai_contrib * sum_by_length[lenB]) % MOD
    
    print(result)

if __name__ == "__main__":
    main()