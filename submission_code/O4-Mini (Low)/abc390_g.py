import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    M = 998244353

    N = int(sys.stdin.readline().strip())

    # Precompute digit‐lengths and 10^d mod M
    d = [0]*(N+1)
    w = [0]*(N+1)
    for i in range(1, N+1):
        di = len(str(i))
        d[i] = di
        w[i] = pow(10, di, M)

    # Precompute factorials up to N and inverses
    fact = [1]*(N+1)
    invfact = [1]*(N+1)
    for i in range(1, N+1):
        fact[i] = fact[i-1]*i % M
    invfact[N] = pow(fact[N], M-2, M)
    for i in range(N, 0, -1):
        invfact[i-1] = invfact[i]*i % M

    # We will compute E[k] = elementary symmetric sum of w[1..N] of degree k,
    # but multiplied by k! to get the sum over ordered k‐tuples:
    #   A[k] = k! * sum_{|S|=k} prod_{j in S} w[j].
    # Then A satisfies the recurrence via generating‐function multiplication.
    #
    # Concretely, build poly G(x) = \prod_{i=1..N} (1 + w[i] x).
    # We need A[k] = k! * [x^k] G(x).  Then for each i we will use A to
    # compute T_i = sum_{k=0..N-1} A_excl_i[k] * (N-1-k)!  where A_excl_i
    # is the same but omitting factor (1 + w[i] x).
    #
    # Key trick:  sum_{k=0..N-1} A_excl_i[k] * (N-1-k)!
    # = [x^<=N-1] G(x)/(1 + w[i] x)  convolved with H(x)=sum (N-1-k)! x^k.
    # We can do one convolution to get F(x)=G(x)*H(x) truncated to x^N,
    # and then evaluate for each i by polynomial division by (1 + w[i] x)
    # evaluated at x=0...x^{N-1}.  That is O(N^2) in general, too big.
    #
    # BUT here N≤2e5, we instead note: (N-1)! times that sum is exactly
    #  sum_{k=0..N-1} [x^k] G(x)/(1 + w[i] x)  * (N-1)!*(N-1-k)!.
    # And   (N-1)!*(N-1-k)! = (N-1)! / k!  * (N-1)!  up to a shift.
    #
    # Actually there's a direct closed‐form by symmetry:
    #
    # sum_{P} f(P)
    # = sum_{i=1..N} i * (N-1)! * [coefficient of x^{<=N-1} in G(x)/(1 + w[i]x)
    #                             evaluated at x=1 ]
    #
    # = (N-1)! * sum_{i=1..N} i * G(1)/(1 + w[i])
    #
    # where G(1) = \prod_{j=1..N} (1 + w[j]) mod M.
    #
    # This formula can be derived by grouping each i in each position,
    # counting the contribution of all ways to choose and order the
    # elements after i (as the expansion of G(x)/(1+w[i]x) at x=1), and
    # the elements before i (the factorial) and summing over positions.
    #
    # We'll implement that:
    total_prod = 1
    for i in range(1, N+1):
        total_prod = total_prod * (1 + w[i]) % M

    inv = pow
    # Precompute inverses of (1 + w[i])
    inv1pw = [0]*(N+1)
    for i in range(1, N+1):
        inv1pw[i] = pow(1 + w[i], M-2, M)

    t = 0
    # sum i * inv(1+w[i])
    for i in range(1, N+1):
        t = (t + i * inv1pw[i]) % M

    ans = fact[N-1] * total_prod % M * t % M
    print(ans)

if __name__ == "__main__":
    main()