import sys

# It's recommended to use fast I/O in Python for competitive programming
# sys.stdin = open('in.txt', 'r')
# input = sys.stdin.readline

def main():
    W, H, L, R, D, U = map(int, sys.stdin.readline().split())
    MOD = 998244353

    MAX_FACT = W + H + 5
    fact = [1] * (MAX_FACT + 1)
    inv_fact = [1] * (MAX_FACT + 1)
    for i in range(1, MAX_FACT + 1):
        fact[i] = (fact[i - 1] * i) % MOD

    inv_fact[MAX_FACT] = pow(fact[MAX_FACT], MOD - 2, MOD)
    for i in range(MAX_FACT - 1, -1, -1):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD

    def nCr_mod(n, r):
        if r < 0 or r > n:
            return 0
        num = fact[n]
        den = (inv_fact[r] * inv_fact[n - r]) % MOD
        return (num * den) % MOD

    # G(w, h) is the number of paths ending at (w, h) on a full grid,
    # which is sum_{i=0..w, j=0..h} C(i+j, i)
    # This simplifies to C(w+h+2, w+1) - 1
    memo_g = {}
    def g(w, h):
        if w < 0 or h < 0:
            return 0
        if (w,h) in memo_g:
            return memo_g[(w,h)]
        
        res = (nCr_mod(w + h + 2, w + 1) - 1 + MOD) % MOD
        memo_g[(w,h)] = res
        return res

    # Total paths on a full grid
    # This is sum_{x=0..W, y=0..H} g(x,y)
    # which is C(W+H+4, W+1) - 1 - (W+1)*(H+1)
    total_full = (nCr_mod(W + H + 4, W + 1) - 1 - (W + 1) * (H + 1)) % MOD
    
    # Calculate total invalid paths (Delta)
    # Invalid paths are those that pass through the hole.
    # We sum over all entry points `p` to the hole:
    # Delta = sum_{p} (paths from any start to p) * (paths from p to any end)
    # paths from start to p = g(p.x, p.y)
    # paths from p to end = g(W-p.x, H-p.y)

    delta = 0
    # Entry from bottom edge of the hole
    for i in range(L, R + 1):
        # Entry point is (i, D)
        # dp_full(i,D) = g(i,D)
        # num_continuations(i,D) = g(W-i, H-D)
        term = (g(i, D) * g(W - i, H - D)) % MOD
        delta = (delta + term) % MOD
    
    # Entry from left edge of the hole (excluding corner (L,D) which is covered above)
    for j in range(D + 1, U + 1):
        # Entry point is (L, j)
        term = (g(L, j) * g(W - L, H - j)) % MOD
        delta = (delta + term) % MOD

    ans = (total_full - delta + MOD) % MOD
    print(ans)

if __name__ == "__main__":
    main()