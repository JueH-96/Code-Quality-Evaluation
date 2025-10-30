import sys
from collections import deque

MOD = 998244353

def main() -> None:
    it = iter(sys.stdin.read().split())
    t = int(next(it))

    ans = []
    for _ in range(t):
        H = int(next(it)); W = int(next(it))
        S = [next(it).strip() for _ in range(H)]

        # Every undirected grid–edge is a node of an ordinary graph G.
        # Indexing: horizontal (H*W) first, then vertical (H*W)
        # id_h(i,j)   – edge between (i,j) and (i,(j+1)%W)
        # id_v(i,j)   – edge between (i,j) and ((i+1)%H,j)
        # Two edges meet in a cell, therefore each cell gives two
        # equality constraints between two ids.

        # We build adjacency lists “if one edge is 1 then the other must be 1”
        # and “if one edge is 0 then the other must be 0” i.e. simple equality.
        nE = 2 * H * W
        g = [[] for _ in range(nE)]          # Undirected equality graph

        def hid(i, j):          # horizontal
            return i * W + j
        def vid(i, j):          # vertical
            return H * W + i * W + j

        for i in range(H):
            for j in range(W):
                hR = hid(i, j)                  # right  edge of this cell
                hL = hid(i, (j-1) % W)          # left   edge
                vD = vid(i, j)                  # down   edge
                vU = vid((i-1) % H, j)          # up     edge

                if S[i][j] == 'A':              # must connect adjacent edges
                    # Four turn patterns → equality between “chosen” state of a
                    # pair of *adjacent* edges and the complement w.r.t the other
                    # pair.  But for equality propagation we only need that
                    # the two edges inside this cell are EQUAL.
                    # For 'A' the chosen two edges are adjacent ⇒ they themselves
                    # must be equal (both 1) and the other two equal (both 0).
                    # Hence two equalities: hR=vU, vU=hR;  hL=vD, vD=hL.
                    # (any of the 4 turns satisfies these)
                    g[hR].append(vU); g[vU].append(hR)
                    g[hL].append(vD); g[vD].append(hL)
                else:                           # 'B' – opposite edges
                    # Chosen pair is opposite, therefore hR==hL , vU==vD
                    g[hR].append(hL); g[hL].append(hR)
                    g[vU].append(vD); g[vD].append(vU)

        colour = [-1] * nE          # 0 : “edge absent”, 1 : “edge present”

        ok = True
        for e in range(nE):
            if colour[e] == -1:
                colour[e] = 0
                dq = deque([e])
                while dq and ok:
                    v = dq.popleft()
                    for nxt in g[v]:
                        if colour[nxt] == -1:
                            colour[nxt] = colour[v]
                            dq.append(nxt)
                        elif colour[nxt] != colour[v]:
                            ok = False
                            break
            if not ok:
                break

        ans.append('0' if not ok else '2')

    sys.stdout.write('
'.join(ans))

if __name__ == '__main__':
    main()