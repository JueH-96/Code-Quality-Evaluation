def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Read input value N (which may be huge but fits in Python's int)
    N = int(data[0])
    mod = 998244353
    
    # Let d be the number of digits in N.
    d = len(str(N))
    
    # We can represent V_N as:
    # V_N = N (10^(d*0)) + N (10^(d*1)) + ... + N (10^(d*(N-1)))
    # This equates to N * (1 + 10^d + 10^(2d) + ... + 10^(d*(N-1))).
    # The sum is a geometric series with ratio r = 10^d.
    # In exact arithmetic:
    # S = (10^(d*N) - 1) / (10^d - 1)
    # So, V_N = N * S.
    #
    # However, since N can be huge (up to 10^18) we must do these calculations modulo 998244353.
    # Let t = 10^d (mod mod)
    #
    # The series sum modulo mod becomes:
    # series = (t^(N) - 1) / (t - 1)  modulo mod.
    # To do this modular division we use the modular inverse.
    
    t = pow(10, d, mod)
    N_mod = N % mod
    
    if t == 1:
        # When t is 1 mod mod the ratio equals 1 and every term is 1.
        # There are N terms, so the sum is simply N mod mod.
        series = N % mod
    else:
        # Calculate geometric series sum:
        # (t^(N) - 1) * inv(t - 1) modulo mod.
        series = (pow(t, N, mod) - 1) * pow(t - 1, mod - 2, mod) % mod
        
    ans = (N_mod * series) % mod
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()