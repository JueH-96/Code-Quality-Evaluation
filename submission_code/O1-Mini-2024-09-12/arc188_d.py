# YOUR CODE HERE
import sys
import math

MOD = 998244353

def main():
    import sys
    import sys
    from math import comb
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    
    used = set(A)
    fixed_B = set()
    for b in B:
        if b != -1:
            fixed_B.add(b)
    if used & fixed_B:
        print(0)
        return
    R = 2*N - N - len(fixed_B)  # Total numbers - A - fixed B
    K = B.count(-1)
    if R < K:
        print(0)
        return
    # Calculate P(R, K) = R! / (R-K)!
    fact = [1]*(2*N+1)
    for i in range(1,2*N+1):
        fact[i] = fact[i-1]*i % MOD
    perm = fact[R] * pow(fact[R-K], MOD-2, MOD) % MOD
    print(perm)

if __name__ == "__main__":
    main()