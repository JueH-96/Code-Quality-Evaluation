def main():
    import sys,sys
    data = sys.stdin.buffer.read().split()
    if not data: 
        return
    it = iter(data)
    try:
        N = int(next(it))
        M = int(next(it))
    except StopIteration:
        return

    # We “encode” each M–element binary sequence as a polynomial
    # f(x)= A₁ + A₂*x + … + A_M*x^(M–1)
    # (thus the first number becomes the constant term).
    #
    # The T–operation is defined by
    #    (T(f))(0)=A₁    and    (T(f))(i)= A₁^A₂^…^A_(i+1)   for i>=1.
    #
    # Two sequences “lie” in the same orbit if one is T^x of the other.
    #
    # In our solution we (a) compute for each sequence its orbit “code” (by finding the
    # canonical — i.e. lexicographically–minimum – member of its T–orbit) and then (b)
    # compute its unique “phase” (the number t so that f = T^t(canonical)).
    #
    # Finally, observe that if two sequences in the same orbit have phases p and q,
    # then f(i,j)= (q - p) mod L (where L is the length of the orbit). In different orbits f=0.
    
    sys.setrecursionlimit(10000000)
    mod = 998244353

    # ----- GF(2) polynomial and T–mapping helper functions -----
    # We represent a binary sequence as an int, with bit i (0-indexed)
    # equal to the ith coefficient (so the first element is the constant term).

    # T_poly does the T–operation (i.e. cumulative XOR of the coefficients).
    def T_poly(f, M):
        running = 0
        res = 0
        i = 0
        while i < M:
            bi = (f >> i) & 1
            running ^= bi
            if running:
                res |= (1 << i)
            i += 1
        return res

    # rev returns the “lex–key” of f (we want to compare the sequence (A₁,…,A_M)
    # with the first element first). Since our polynomial f has A₁ as bit0, we “reverse”
    # the M bits so that the constant term becomes the most–significant.
    def rev(f, M):
        r = 0
        for i in range(M):
            r = (r << 1) | ((f >> i) & 1)
        return r

    # --- GF(2) polynomial operations mod x^d, for small d.
    def gf2_mul(a, b, d):
        res = 0
        i = 0
        while i < d:
            if (b >> i) & 1:
                res ^= (a << i)
            i += 1
        return res & ((1 << d) - 1)
    def gf2_inv(poly, d):
        limit = 1 << d
        for candidate in range(limit):
            if gf2_mul(poly, candidate, d) == 1:
                return candidate
        return None
    def gf2_div(a, b, d):
        inv_b = gf2_inv(b, d)
        return gf2_mul(a, inv_b, d)

    # Precompute a “discrete log” map.
    # In GF2 one may show that 
    #   (1/(1-x))^t = sum_{k>=0} binom(t+k-1, k) * x^k  (with constant=1).
    # For t in [0, R) with R = 2^(ceil_log2(M)) (if M>1) it suffices to know the first d = (M-1).bit_length()+1 coefficients.
    if M > 1:
        d_global = (M - 1).bit_length() + 1
    else:
        d_global = 0
    discrete_log_map = {}
    if d_global:
        bitcount = d_global
        lim = 1 << bitcount
        for t_val in range(lim):
            pat = 1  # constant coefficient always 1.
            for k in range(1, bitcount):
                # Lucas theorem implies binom(t+k-1, k) mod2 = 1 iff ((t+k-1)& k)== k.
                bit_val = 1 if ((t_val + k - 1) & k) == k else 0
                pat = (pat << 1) | bit_val
            discrete_log_map[pat] = t_val

    # get_phase(g, M, can) returns the unique 0<= t < (1<<d_global) such that
    #   g = T^t(can)   (i.e. g = can * (1/(1-x))^t mod x^d_global)
    def get_phase(g, M, can):
        if M <= 1:
            return 0
        d = (M - 1).bit_length() + 1
        mask = (1 << d) - 1
        g0 = g & mask
        can0 = can & mask
        inv_can0 = gf2_inv(can0, d)
        Q = gf2_mul(g0, inv_can0, d)
        return discrete_log_map.get(Q, 0)
    
    # orbit_memo stores for each polynomial f its (can, phase, L)
    orbit_memo = {}
    def get_orbit_info(f, M):
        if f in orbit_memo:
            return orbit_memo[f]
        orbit_list = []
        cur = f
        t = 0
        best = cur
        best_t = 0
        best_key = rev(cur, M)
        while True:
            orbit_list.append(cur)
            cur_key = rev(cur, M)
            if cur_key < best_key:
                best = cur
                best_key = cur_key
                best_t = t
            t += 1
            cur = T_poly(cur, M)
            if cur == f:
                break
        L = t
        can = best
        for idx, val in enumerate(orbit_list):
            if val not in orbit_memo:
                if val == can:
                    ph = 0
                else:
                    if (can & 1) == 0:
                        ph = 0
                    else:
                        ph = get_phase(val, M, can)
                orbit_memo[val] = (can, ph, L)
        return orbit_memo[f]
    
    # Read the N sequences.
    seq_orbit = []
    groups = {}  # group by canonical representative
    for _ in range(N):
        poly = 0
        for i in range(M):
            try:
                bi = int(next(it))
            except StopIteration:
                break
            if bi:
                poly |= (1 << i)
        info = get_orbit_info(poly, M)  # (can, phase, L)
        seq_orbit.append(info)
        can, ph, L_val = info
        groups.setdefault(can, []).append((ph, L_val))
    
    # Now, in each orbit group (all sequences with same canonical representative),
    # if two sequences with phases p and q (in an orbit of length L) occur in the input order,
    # we add f(i,j)= (q - p) mod L.
    # We sum these over all pairs i<j in the input order.
    #
    # We use a Fenw (Binary Indexed Tree) to sum these differences in O(n log n) per group.
    class Fenw:
        __slots__ = ('n','f')
        def __init__(self,n):
            self.n = n
            self.f = [0]*(n+1)
        def update(self,i,delta):
            i += 1
            while i <= self.n:
                self.f[i] += delta
                i += i & -i
        def query(self,i):
            s = 0
            i += 1
            while i:
                s += self.f[i]
                i -= i & -i
            return s
    class FenwSum:
        __slots__ = ('n','f')
        def __init__(self, n):
            self.n = n
            self.f = [0]*(n+1)
        def update(self, i, delta):
            i += 1
            while i <= self.n:
                self.f[i] += delta
                i += i & -i
        def query(self, i):
            s = 0
            i += 1
            while i:
                s += self.f[i]
                i -= i & -i
            return s

    total_ans = 0
    for can, arr in groups.items():
        # All members of one group have the same orbit length.
        L_val = arr[0][1]
        fenw_count = Fenw(L_val)
        fenw_sum = FenwSum(L_val)
        group_ans = 0
        cnt = 0
        sum_ph = 0
        for (p, _) in arr:
            cnt_le = fenw_count.query(p)
            sum_le = fenw_sum.query(p)
            cnt_total = cnt
            cnt_gt = cnt_total - cnt_le
            # For an earlier element with phase a:
            # if a <= p, contribution is (p - a)
            # if a > p, contribution is (L_val - a + p)
            contrib = p * cnt_le - sum_le + (L_val + p) * cnt_gt - (sum_ph - sum_le)
            group_ans += contrib
            fenw_count.update(p, 1)
            fenw_sum.update(p, p)
            cnt += 1
            sum_ph += p
        total_ans = (total_ans + group_ans) % mod

    sys.stdout.write(str(total_ans))

if __name__ == '__main__':
    main()