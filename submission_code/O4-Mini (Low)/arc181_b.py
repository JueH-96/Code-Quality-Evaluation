import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    
    # we'll use a simple rolling hash on S to compare substrings in O(1)
    MOD = (1<<61) - 1
    def mix(a,b):
        c = a*b
        c = (c >> 61) + (c & MOD)
        if c >= MOD: c -= MOD
        return c
    def add(a,b):
        c = a + b
        if c >= MOD: c -= MOD
        return c
    
    t = int(input())
    out = []
    for _ in range(t):
        S = input().strip()
        X = input().strip()
        Y = input().strip()
        n0x = X.count('0')
        n1x = len(X) - n0x
        n0y = Y.count('0')
        n1y = len(Y) - n0y
        lenS = len(S)
        
        # length equation: lenS*n0x + lenT*n1x == lenS*n0y + lenT*n1y
        # => lenT * (n1x - n1y) = lenS*(n0y - n0x)
        dx = n1x - n1y
        dy = n0y - n0x
        if dx == 0:
            # then must have dy == 0 or no solution
            if dy != 0:
                out.append("No")
                continue
            # any T length works; no further constraint => Yes
            out.append("Yes")
            continue
        # else lenT = lenS*dy/dx, must be non-negative integer
        if dy * dx < 0:
            out.append("No")
            continue
        if (lenS * dy) % dx != 0:
            out.append("No")
            continue
        lenT = (lenS * dy) // dx
        
        # precompute hash of S
        n = lenS
        H = [0]*(n+1)
        P = [1]*(n+1)
        base = 1315423911  # random
        for i,ch in enumerate(S):
            H[i+1] = add(mix(H[i], base), ord(ch))
            P[i+1] = mix(P[i], base)
        def get_hash(l, r):
            # S[l:r], 0-based, r exclusive
            res = H[r] - mix(H[l], P[r-l])
            if res < 0: res += MOD
            return res
        
        # forced assignments of T: forced[i] = char or None
        forced = [None]*lenT
        
        # two pointers over X and Y segments
        ix = 0
        iy = 0
        # how many chars remain in current segment
        remx = lenS if ix < len(X) and X[ix]=='0' else lenT if ix < len(X) else 0
        remy = lenS if iy < len(Y) and Y[iy]=='0' else lenT if iy < len(Y) else 0
        # positions inside S or T for each side
        ox = 0  # offset into current segment
        oy = 0
        
        ok = True
        while ix < len(X) and iy < len(Y) and ok:
            d = min(remx, remy)
            tx = X[ix]
            ty = Y[iy]
            # case by types
            if tx == '0' and ty == '0':
                # both in S
                # compare S[ox : ox+d] vs S[oy : oy+d]
                # but segments may wrap, so we do in at most two parts each
                # we'll compare piecewise
                a_st = ox
                b_st = oy
                left = d
                # break into possibly two runs if we wrap
                while left>0 and ok:
                    a_run = min(left, n - (a_st % n))
                    b_run = min(left, n - (b_st % n))
                    length = min(a_run, b_run)
                    if get_hash(a_st % n, (a_st % n)+length) != get_hash(b_st % n, (b_st % n)+length):
                        ok = False
                        break
                    left -= length
                    a_st += length
                    b_st += length
            elif tx == '1' and ty == '1':
                # T vs T: no constraint
                pass
            else:
                # one side S, one side T
                # we want to map the T positions to S positions
                if tx == '0':
                    # X's S vs Y's T
                    s_off = ox
                    t_off = oy
                else:
                    # X's T vs Y's S
                    s_off = oy
                    t_off = ox
                # now for k in [0..d):
                for k in range(d):
                    ch = S[(s_off+k) % n]
                    pos = t_off + k
                    if forced[pos] is None:
                        forced[pos] = ch
                    elif forced[pos] != ch:
                        ok = False
                        break
            # advance
            remx -= d
            remy -= d
            ox += d
            oy += d
            if remx == 0:
                ix += 1
                ox = 0
                if ix < len(X):
                    remx = lenS if X[ix]=='0' else lenT
            if remy == 0:
                iy += 1
                oy = 0
                if iy < len(Y):
                    remy = lenS if Y[iy]=='0' else lenT
        
        out.append("Yes" if ok else "No")
    
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()