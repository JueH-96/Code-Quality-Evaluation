# YOUR CODE HERE
def main():
    import sys,sys
    data = sys.stdin.read().strip().split()
    if not data: 
        return
    try:
        W = int(data[0])
        H = int(data[1])
        L = int(data[2])
        R = int(data[3])
        D = int(data[4])
        U = int(data[5])
    except:
        return

    MOD = 998244353
    # We'll need factorials up to (W+H+~10); note that W, H <= 10^6.
    Nmax = W + H + 10
    fact = [1]*(Nmax+1)
    invfact = [1]*(Nmax+1)
    for i in range(2, Nmax+1):
        fact[i] = fact[i-1]*i % MOD
    invfact[Nmax] = pow(fact[Nmax], MOD-2, MOD)
    for i in range(Nmax, 0, -1):
        invfact[i-1] = invfact[i]*i % MOD

    def nCr(n, r):
        if r < 0 or r > n:
            return 0
        return fact[n]*invfact[r] % MOD * invfact[n-r] % MOD

    # f_F(x,y): the number of paths starting at (x,y) if every point in 
    # {0<=x<=W, 0<=y<=H} is allowed.
    # One may show: f_F(x,y) = Sum_{a=0}^{W-x} Sum_{b=0}^{H-y} binom(a+b, a)
    #                   = nCr((W-x)+(H-y)+2, (W-x)+1) - 1.
    def F_full(x, y):
        dx = W - x
        dy = H - y
        return (nCr(dx+dy+2, dx+1) - 1) % MOD

    # F_sum(P,Q): for nonnegative P,Q, we want S = sum_{u=0}^P sum_{v=0}^Q ( nCr(u+v+2, u+1) - 1 ).
    # A short computation (using hockey–stick) shows that:
    #       F_sum(P,Q) = nCr(P+Q+4, P+2) - (P+Q+4) - (P+1)*(Q+1).
    def F_sum(P, Q):
        if P < 0 or Q < 0:
            return 0
        res = nCr(P+Q+4, P+2) - (P+Q+4) - (P+1)*(Q+1)
        return res % MOD

    # Total count (over every starting block in the full grid)
    # T_F = nCr(W+H+4, H+2) - (H+3) - (W+1) - (W+1)*(H+1)
    T_F = (nCr(W+H+4, H+2) - (H+3) - (W+1) - (W+1)*(H+1)) % MOD

    # Next we count paths that eventually hit (or start in) the forbidden rectangle H0:
    #   H0 = {(x,y) : L<= x <= R,  D<= y <= U }
    #
    # (a) Count those that “first hit” H0 along its boundary.
    # Note that any first–hit must occur at p with either p.x=L (vertical boundary)
    # or p.y=D (horizontal boundary). (The block (L,D) can be entered in two ways.)
    # Define G(p) = number of ways (from the “safe region” where
    #      safe = {(x,y) in F : not(x>=L and y>=D)}
    # to approach p so that the next step enters p.
    # Then (for p = (L,y)):
    #  – if L>0 and y > D, only valid predecessor is (L–1,y), and there are
    #       nCr(L+y+1, L) – 1   ways.
    #  – if p = (L,D) and L>0, D>0, then G((L,D)) = (nCr(L+D+1, L) – 1) + (nCr(L+D+1, L+1) – 1).
    #  – if L==0 (or D==0) we define G(p)=1.
    #
    # Then the number of paths that "first hit" p is  G(p)*F_full(p.x, p.y).
    #
    # Sum over p in the two boundaries:
    S_bad_boundary = 0
    # Vertical boundary: p = (L,y) for y from D to U.
    M_term = 0
    if D <= U:
        if L == 0:
            g_val = 1
        elif D == 0:
            g_val = 1
        else:
            g_val = (nCr(L+D+1, L) - 1 + nCr(L+D+1, L+1) - 1) % MOD
        M_term = g_val * F_full(L, D) % MOD
    V_sum = 0
    for y in range(D+1, U+1):
        if L == 0:
            g_val = 1
        else:
            g_val = (nCr(L+y+1, L) - 1) % MOD
        V_sum = (V_sum + g_val * F_full(L, y)) % MOD
    H_sum = 0
    for x in range(L+1, R+1):
        if D == 0:
            h_val = 1
        else:
            h_val = (nCr(x+D+1, x+1) - 1) % MOD
        H_sum = (H_sum + h_val * F_full(x, D)) % MOD
    S_bad_boundary = (M_term + V_sum + H_sum) % MOD

    # (b) Count those paths whose starting block is in H0.
    # That is, S_start_H0 = sum_{x=L}^{R} sum_{y=D}^{U} F_full(x,y).
    # Change variables u = W – x, v = H – y.
    # Then x runs from L to R <=> u runs from W–R to W–L,
    #      and y from D to U <=> v runs from H–U to H–D.
    A_val = W - R
    B_val = W - L
    C_val = H - U
    D_val = H - D
    S_start_H0 = ( F_sum(B_val, D_val)
                   - F_sum(A_val - 1, D_val)
                   - F_sum(B_val, C_val - 1)
                   + F_sum(A_val - 1, C_val - 1) ) % MOD

    total_bad = (S_bad_boundary + S_start_H0) % MOD

    valid = (T_F - total_bad) % MOD
    sys.stdout.write(str(valid))
    
if __name__ == '__main__':
    main()