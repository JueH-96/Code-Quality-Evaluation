MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    A = [0] + A  # A[1] to A[N] are valid
    
    if N == 0:
        print(0)
        return
    
    inv_N = pow(N, MOD-2, MOD)
    K = ( ( (N + 1) * inv_N ) % MOD + MOD ) % MOD
    
    s = [0] * (N + 2)  # s[0..N]
    
    for x in range(N-1, -1, -1):
        # Compute s[x] = (A[x+1] + K * s[x+1]) % MOD
        term = (A[x+1] + (K * s[x+1]) % MOD) % MOD
        s[x] = term
    
    ans = (s[0] * inv_N) % MOD
    print(ans)
    
if __name__ == "__main__":
    main()