def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    mod = 998244353
    
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # Precompute prefix sums (modulo) for A
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i+1] = (prefix_sum[i] + A[i]) % mod

    # Precompute powers of 10 up to 10^10 (we need up to 10 digits for numbers up to 10^9)
    max_len = 10  # Because 10^9 has at most 10 digits
    pow10 = [1] * (max_len + 1)
    for i in range(1, max_len + 1):
        pow10[i] = (pow10[i-1] * 10) % mod

    # Calculate the result
    result = 0
    for j in range(1, N):
        # Number of digits in A[j]
        length = len(str(A[j]))
        # sum_{i=0..j-1} [ A[i] * 10^length ] = prefix_sum[j] * 10^length
        # plus (j times) A[j] for the f(A_i, A_j) summation
        part = (prefix_sum[j] * pow10[length] + (j * A[j]) ) % mod
        result = (result + part) % mod

    print(result % mod)

if __name__ == "__main__":
    main()