def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    mod = 998244353

    # Precompute powers of 10 up to 10 digits (since A_i <= 1e9 has at most 10 digits)
    pow10 = [1] * 11
    for k in range(1, 11):
        pow10[k] = pow10[k-1] * 10 % mod

    ans = 0
    prefix_sum = 0  # sum of A[0..j-1]
    for j, x in enumerate(A, start=1):
        # number of digits in x
        # integer math faster than str()
        t = x
        length = 0
        while t > 0:
            length += 1
            t //= 10
        # contribution from concat(A_i, x) for i < j: sum_i (A_i * 10^length)
        ans = (ans + prefix_sum * pow10[length]) % mod
        # contribution from concat(A_i, x): sum_i x = x * (number of i's) = x*(j-1)
        ans = (ans + x * (j-1)) % mod

        prefix_sum = (prefix_sum + x) % mod

    print(ans)

if __name__ == "__main__":
    main()