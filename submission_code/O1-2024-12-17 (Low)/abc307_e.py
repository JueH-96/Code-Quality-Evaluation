def main():
    import sys
    mod = 998244353

    data = sys.stdin.read().strip().split()
    N, M = map(int, data)

    # If M=1, no two adjacent persons can have the same integer => impossible if N>=2
    if M == 1:
        print(0)
        return

    # Fast exponentiation: (M-1)^N mod 998244353
    pow_term = pow(M-1, N, mod)

    # Add the correction term: (M-1)*(-1)^N
    #    If N is even, we add (M-1)
    #    If N is odd,  we subtract (M-1)
    if N % 2 == 0:
        ans = (pow_term + (M-1)) % mod
    else:
        ans = (pow_term - (M-1)) % mod

    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()