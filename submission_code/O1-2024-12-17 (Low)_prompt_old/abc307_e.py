def solve():
    import sys
    mod = 998244353
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    
    # If M == 1, only one integer is available. 
    # For N >= 2 in a circle, at least two are adjacent with the same integer.
    # Thus, no valid assignments.
    if M == 1:
        print(0)
        return
    
    # Compute (M - 1)^N modulo 998244353
    pow_m1 = pow(M - 1, N, mod)
    
    # Compute the term (M-1)*(-1)^N in modulo
    # (-1)^N is 1 if N is even, (mod - 1) if N is odd (because -1 ≡ mod-1).
    if N % 2 == 0:
        sign = 1
    else:
        sign = mod - 1
    
    # Total = (M - 1)^N + (M - 1)*((-1)^N)  mod 998244353
    result = (pow_m1 + (M - 1) * sign) % mod
    
    print(result)

def main():
    solve()

if __name__ == "__main__":
    main()