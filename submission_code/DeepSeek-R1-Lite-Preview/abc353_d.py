def main():
    MOD = 998244353
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Precompute power_of_10 for d from 0 to 10
    power_of_10 = [1]  # d = 0
    for d in range(1, 11):
        power_of_10.append((power_of_10[d - 1] * 10) % MOD)
    
    # Compute d_j for each A_j
    d = [len(str(a)) for a in A]
    
    # Compute prefix_sum
    prefix_sum = [0] * N
    for j in range(1, N):
        prefix_sum[j] = (prefix_sum[j - 1] + A[j - 1]) % MOD
    
    # Compute S1 and S2
    S1 = 0
    S2 = 0
    for j in range(1, N + 1):
        if j - 1 >= 0:
            S1 = (S1 + prefix_sum[j - 1] * power_of_10[d[j - 1]]) % MOD
            S2 = (S2 + (j - 1) * A[j - 1]) % MOD
    
    # Total sum
    total_sum = (S1 + S2) % MOD
    print(total_sum)

if __name__ == "__main__":
    main()