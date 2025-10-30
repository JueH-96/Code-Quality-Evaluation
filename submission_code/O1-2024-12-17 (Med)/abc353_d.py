def main():
    import sys

    MOD = 998244353
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # Precompute powers of 10 up to 10 digits (since 1 <= A_i <= 10^9, max 10 digits)
    pow10 = [1] * 11
    for i in range(1, 11):
        pow10[i] = (pow10[i-1] * 10) % MOD
    
    # Function to count decimal digits of a positive integer
    def digits(x):
        return len(str(x))
    
    # Compute prefix sums of A (mod MOD)
    prefix = [0] * (N+1)
    for i in range(1, N+1):
        prefix[i] = (prefix[i-1] + A[i-1]) % MOD
    
    # Calculate the sum of f(A_i, A_j) for all 1 <= i < j <= N
    # f(x,y) = x * 10^(digits(y)) + y
    # Summation rearranged:
    #   sum_{j=2..N} [ 10^(digits(A_j)) * (sum of A_i for i=1..j-1) + (j-1) * A_j ]
    ans = 0
    for j in range(2, N+1):
        length_j = digits(A[j-1])
        sum_i = prefix[j-1]  # sum of A[0..j-2]
        contrib = (sum_i * pow10[length_j]) % MOD
        contrib = (contrib + (j-1) * A[j-1]) % MOD
        ans = (ans + contrib) % MOD
    
    print(ans)

# Don't forget to call main().
main()