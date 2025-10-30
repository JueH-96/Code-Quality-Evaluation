# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Calculate prefix sums
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + A[i]
    
    # Calculate expected value as a fraction
    numerator = 0
    denominator = N
    
    for k in range(N):
        # Sum of A[i] for i > k
        sum_A_greater_k = prefix_sum[N] - prefix_sum[k]
        numerator += sum_A_greater_k
    
    # numerator / denominator is the expected value
    # We need to find numerator * denominator^(-1) % MOD
    
    # Function to compute modular inverse
    def mod_inv(x, mod):
        return pow(x, mod - 2, mod)
    
    denominator_inv = mod_inv(denominator, MOD)
    
    # Calculate the result
    result = (numerator * denominator_inv) % MOD
    
    print(result)

main()