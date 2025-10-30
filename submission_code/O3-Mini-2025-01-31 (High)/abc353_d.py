def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:]))
    MOD = 998244353

    # Precompute powers of 10 for digit lengths 1 to 11 (since numbers up to 10^9 have at most 10 digits)
    pow10 = [1] * 12
    for i in range(1, 12):
        pow10[i] = (pow10[i - 1] * 10) % MOD

    # Utility to compute number of digits of x (x is positive)
    def num_digits(x):
        if x < 10:
            return 1
        elif x < 100:
            return 2
        elif x < 1000:
            return 3
        elif x < 10000:
            return 4
        elif x < 100000:
            return 5
        elif x < 1000000:
            return 6
        elif x < 10000000:
            return 7
        elif x < 100000000:
            return 8
        elif x < 1000000000:
            return 9
        else:
            return 10

    result = 0
    prefix_sum = 0  # Sum of A[0] through A[j-1]
    # Iterate over indices treating each A[j] as the second number in the pair f(A_i, A_j)
    for j in range(n):
        a = A[j]
        if j > 0:
            # Every pair (i,j) with i < j contributes two parts:
            # 1. A_i is "shifted": multiplied by 10^(digits of A[j]); summing over i gives prefix_sum * 10^(digits(A[j]))
            # 2. A_j is simply added for every i where i<j: that happens j times.
            result = (result + prefix_sum * pow10[num_digits(a)]) % MOD
            result = (result + a * j) % MOD
        prefix_sum = (prefix_sum + a) % MOD

    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()