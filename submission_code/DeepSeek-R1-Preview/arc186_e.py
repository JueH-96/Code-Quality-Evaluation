MOD = 998244353

def main():
    import sys
    N, M, K = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))
    
    # Check if all elements in X are the same
    c = X[0]
    for x in X[1:]:
        if x != c:
            print(0)
            return
    
    # Now, X is composed of M copies of c
    # We need sequences with exactly M-1 c's and all other K-1 characters appearing at least once
    
    # If N < M-1, it's impossible
    if M-1 > N:
        print(0)
        return
    
    # The number of ways to choose positions for c's
    from math import comb
    choose = comb(N, M-1) % MOD
    
    # The remaining positions: rem = N - (M-1)
    rem = N - (M-1)
    # The remaining positions must be filled with at least one of each of the K-1 other characters
    if rem < K-1:
        print(0)
        return
    
    # Compute the number of ways to fill 'rem' positions with K-1 characters, each at least once
    # Using inclusion-exclusion
    k = K - 1
    res = 0
    for i in range(k+1):
        term = pow(k - i, rem, MOD)
        term = term * comb(k, i) % MOD
        if i % 2 == 0:
            res = (res + term) % MOD
        else:
            res = (res - term) % MOD
    
    total = choose * res % MOD
    print(total)

if __name__ == '__main__':
    main()