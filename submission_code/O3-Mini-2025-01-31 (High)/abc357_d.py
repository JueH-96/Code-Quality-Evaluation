def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Read N as a string to easily compute its number of digits.
    N_str = data[0]
    N = int(N_str)
    
    mod = 998244353
    # Let d be the number of digits in N.
    d = len(N_str)
    
    # The integer V_N is formed by concatenating N exactly N times.
    # If we let A = N and note that N (as a number) has d digits,
    # then V_N can be written as:
    #   V_N = N * (10^(d*(N-1)) + 10^(d*(N-2)) + ... + 10^0)
    # which is N multiplied by the geometric series:
    #   S = 1 + 10^d + 10^(2*d) + ... + 10^{d*(N-1)}.
    #
    # Let r = 10^d mod mod.
    # Then S = 1 + r + r^2 + ... + r^(N-1).
    # The sum of a geometric series is S = (r^N - 1)/(r - 1) if r != 1.
    #
    # Compute these values using modular arithmetic.
    
    A = N % mod
    r = pow(10, d, mod)
    
    if r == 1:
        # When 10^d ≡ 1 mod mod, each term of the series is 1.
        S = N % mod
    else:
        # Use the formula for a geometric series modulo mod.
        S = (pow(r, N, mod) - 1) * pow(r - 1, mod - 2, mod) % mod

    result = (A * S) % mod
    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()