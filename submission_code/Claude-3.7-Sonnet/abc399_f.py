def main():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    MOD = 998244353
    result = 0
    
    for l in range(N):
        subarray_sum = 0
        for r in range(l, N):
            subarray_sum = (subarray_sum + A[r]) % MOD
            result = (result + pow(subarray_sum, K, MOD)) % MOD
    
    print(result)

if __name__ == "__main__":
    main()