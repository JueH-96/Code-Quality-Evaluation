def solve():
    import sys

    MOD = 998244353
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # Precompute powers of 10 mod MOD up to 10^10 (since max number of digits of A_i <= 10)
    max_len = 10  # max digits for 1 <= A_i <= 10^9
    pow10 = [1] * (max_len + 1)
    for i in range(1, max_len + 1):
        pow10[i] = (pow10[i - 1] * 10) % MOD

    # Function to compute the number of digits
    def num_digits(x):
        return len(str(x))

    # Precompute prefix sums of A
    prefix_sums = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_sums[i] = (prefix_sums[i - 1] + A[i - 1]) % MOD

    # Calculate result
    result = 0
    for j in range(2, N + 1):
        # length of A_j
        length_j = num_digits(A[j - 1])
        # Add contribution: 10^len(A_j) * sum(A_i for i < j)
        result += pow10[length_j] * prefix_sums[j - 1]
        result %= MOD
        # Add contribution: A_j * (j-1)
        result += A[j - 1] * (j - 1)
        result %= MOD

    print(result % MOD)

# Call solve()
solve()