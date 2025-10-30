def main():
    import sys
    
    MOD = 998244353
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Precompute powers of 10 up to 10 digits (since 10^9 can have up to 10 digits)
    p10 = [1] * 11
    for i in range(1, 11):
        p10[i] = (p10[i - 1] * 10) % MOD

    # Prefix sums of A (modulo)
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = (prefix_sum[i] + A[i]) % MOD

    # Function to get the number of digits of x (x >= 1)
    def digits(x):
        return len(str(x))
    
    answer = 0
    for j in range(1, N):
        d = digits(A[j])
        # sum_{i=0..j-1} f(A[i], A[j]) = sum_{i=0..j-1} [A[i]*10^d + A[j]]
        # = (10^d)*sum_{i=0..j-1} A[i] + j*A[j]
        part = prefix_sum[j] * p10[d]
        part += (j * A[j])
        part %= MOD
        answer = (answer + part) % MOD

    print(answer % MOD)

# Do not forget to call main() at the end
main()