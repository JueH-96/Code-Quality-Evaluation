import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    from math import gcd
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    P = [int(next(it)) - 1 for _ in range(N)]
    A = [int(next(it)) for _ in range(N)]
    # Build cycles
    vis = [False]*N
    # For each index: store cycle_id and index in its cycle
    idx_in_cycle = [0]*N
    cycle_id_of = [0]*N
    cycles = []  # list of tuples (first_pos, cycle_index_list, s_list, idx_first)
    for i in range(N):
        if not vis[i]:
            c = []
            x = i
            while not vis[x]:
                vis[x] = True
                c.append(x)
                x = P[x]
            # c now is one cycle in order c[0]->c[1]->...
            d = len(c)
            # s values
            s = [A[pos] for pos in c]
            # find first_pos (smallest index) and its idx
            minpos = c[0]
            minidx = 0
            for j,pos in enumerate(c):
                if pos < minpos:
                    minpos = pos
                    minidx = j
            # record idx_in_cycle and cycle_id
            cid = len(cycles)
            for j,pos in enumerate(c):
                cycle_id_of[pos] = cid
                idx_in_cycle[pos] = j
            cycles.append((minpos, c, s, minidx))
    # Sort cycles by first_pos
    cycles.sort(key=lambda x: x[0])
    # Prepare answer
    ans = [0]*N
    # Current constraint k ≡ K mod M
    K = 0
    M = 1
    # Extended gcd for modular inverse
    def extgcd(a,b):
        if b==0:
            return (a,1,0)
        g,x1,y1 = extgcd(b, a%b)
        # g = b*x1 + (a%b)*y1 = b*x1 + (a - (a//b)*b)*y1 = a*y1 + b*(x1 - (a//b)*y1)
        return (g, y1, x1 - (a//b)*y1)
    def modinv(a, m):
        # inverse of a mod m, m>0, gcd(a,m)=1
        g, x, y = extgcd(a, m)
        # x*a + y*m = g = 1
        return x % m
    # Process each cycle
    for firstpos, c, s, idx_first in cycles:
        d = len(c)
        # compute gcd with current M
        g = gcd(M, d)
        # residue mod g that k must satisfy
        r0 = K % g
        # scan residues r = r0 + t*g for t in [0, d/g)
        # and pick r that minimizes s[(idx_first + r) % d]
        best_val = None
        best_r = None
        # number of candidates = d//g
        step = g
        # compute start r0 in [0,d)
        r0_mod = r0 % d
        # We want all r ≡ r0_mod (mod g), so r = r0_mod + t*g
        # but ensure r<d
        # t from 0 to (d - r0_mod + g -1)//g but we know d%g==0 so count = d//g
        cnt = d // g
        r = r0_mod
        # loop t times
        for _ in range(cnt):
            idx = idx_first + r
            # mod d
            if idx >= d:
                idx %= d
            val = s[idx]
            if best_val is None or val < best_val:
                best_val = val
                best_r = r
            # if equal val, we can keep earlier best_r (smaller r) to break tie
            r += step
            if r >= d:
                r %= d
        r = best_r  # chosen rotation for this cycle
        # Merge k ≡ K mod M and k ≡ r mod d
        # Solve M * t + K ≡ r (mod d) => M*t ≡ (r-K) mod d
        # Let g = gcd(M,d) (we have this), and ensure (r-K)%g==0 by construction
        # Then reduce: M'=M/g, d'=d/g
        M1 = M // g
        d1 = d // g
        # Compute (r-K)//g mod d1
        rhs = (r - K) // g % d1
        # Compute inverse of M1 mod d1
        invM1 = modinv(M1 % d1, d1)
        t0 = (rhs * invM1) % d1
        # new K
        K = K + M * t0
        # new modulus
        M = M * d1  # lcm
        # reduce K modulo M
        K %= M
        # Fill answer for this cycle: ans[c[j]] = s[(j + r)%d]
        for j, pos in enumerate(c):
            jj = j + r
            if jj >= d:
                jj %= d
            ans[pos] = s[jj]
    # print answer
    out = sys.stdout
    out.write(" ".join(str(ans[i]) for i in range(N)))
    out.write("
")

if __name__ == "__main__":
    main()