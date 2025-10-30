def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # Read N as an integer.
    N = int(data[0])
    mod = 998244353
    
    # Determine the number of digits of N.
    s = str(N)
    d = len(s)
    
    # In V_N, the integer is obtained by concatenating N exactly N times.
    # This value is: V_N = N * (10^(d*(N-1)) + 10^(d*(N-2)) + ... + 1)
    # This sum is a geometric series with ratio r = 10^d.
    # Hence, V_N = N * (r^N - 1) / (r - 1).
    
    # Compute r = 10^d modulo mod.
    r = pow(10, d, mod)
    
    # Compute the geometric series sum, S = (r^N - 1) / (r - 1) modulo mod.
    # Using modular arithmetic, we compute the numerator and then multiply by 
    # the modular inverse of (r - 1).
    numerator = (pow(r, N, mod) - 1) % mod
    # Compute the modular inverse using Fermat's little theorem, which works because mod is prime.
    inv_denominator = pow(r - 1, mod - 2, mod)
    S = (numerator * inv_denominator) % mod
    
    # Multiply N and S modulo mod to get the result.
    result = (N % mod) * S % mod
    
    # Write the result to stdout.
    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()