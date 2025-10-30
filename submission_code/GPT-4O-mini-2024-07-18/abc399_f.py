def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:N+2]))
    
    MOD = 998244353
    
    # Precompute prefix sums
    prefix_sum = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_sum[i] = (prefix_sum[i - 1] + A[i - 1]) % MOD
    
    total_sum = 0
    
    # Calculate the sum of (sum(A[l:r]))^K for all 1 <= l <= r <= N
    for l in range(1, N + 1):
        for r in range(l, N + 1):
            segment_sum = (prefix_sum[r] - prefix_sum[l - 1]) % MOD
            total_sum = (total_sum + pow(segment_sum, K, MOD)) % MOD
    
    print(total_sum)

if __name__ == "__main__":
    main()