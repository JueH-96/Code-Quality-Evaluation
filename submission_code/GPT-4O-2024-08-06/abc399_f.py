# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    MOD = 998244353
    
    # Compute prefix sums
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P[i] + A[i]
    
    result = 0
    
    # Iterate over all subarrays
    for l in range(N):
        for r in range(l, N):
            subarray_sum = P[r+1] - P[l]
            result = (result + pow(subarray_sum, K, MOD)) % MOD
    
    print(result)

if __name__ == "__main__":
    main()