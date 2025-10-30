import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    MOD = 998244353

    W, H, L, R, D, U = map(int, sys.stdin.readline().split())

    # Precompute factorials and inverse factorials up to W+H+5
    N = W + H + 5
    fact = [1] * (N)
    ifact = [1] * (N)
    for i in range(1, N):
        fact[i] = fact[i-1] * i % MOD
    # Fermat inverse of factorial
    ifact[N-1] = pow(fact[N-1], MOD-2, MOD)
    for i in range(N-1, 0, -1):
        ifact[i-1] = ifact[i] * i % MOD

    def comb(n, k):
        if k < 0 or k > n:
            return 0
        return fact[n] * ifact[k] % MOD * ifact[n-k] % MOD

    # Sum of binomial paths in an x by y grid from (0,0) to any (i,j):
    # S(x,y) = sum_{i=0..x, j=0..y} C(i+j, i)
    # It is known S(x,y) = C(x+y+2, x+1) - 1
    def S(x, y):
        if x < 0 or y < 0:
            return 0
        return (comb(x + y + 2, x + 1) - 1) % MOD

    # Total number of monotonic up/right paths starting anywhere to anywhere:
    total = S(W, H)

    # Count paths that first touch the forbidden rectangle from the left side.
    # Entry is via (L-1, y) for y in [D..U].  Sum over y of:
    #   ways to get to (L-1, y) times ways from (L-1, y) to any end.
    bad = 0
    for y in range(D, U+1):
        to_entry = S(L-1, y)
        # from entry to ends in the remaining region of width W-(L-1), height H-y
        from_entry = S(W - (L-1), H - y)
        bad = (bad + to_entry * from_entry) % MOD

    # Similarly from the bottom side: entry via (x, D-1), x in [L..R].
    for x in range(L, R+1):
        to_entry = S(x, D-1)
        from_entry = S(W - x, H - (D-1))
        bad = (bad + to_entry * from_entry) % MOD

    ans = (total - bad) % MOD
    print(ans)

if __name__ == "__main__":
    main()