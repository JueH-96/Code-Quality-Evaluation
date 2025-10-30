def main():
    import sys
    from array import array
    input = sys.stdin.readline

    H, W, Q = map(int, input().split())
    N = H * W
    sentinel = N

    # Four "next" arrays for skipping removed cells in each direction.
    # We use array('i') so each entry is a 32-bit C-int, saving memory.
    # Indices 0..N-1 correspond to grid cells, index N is a sentinel.
    nxt_r = array('i', range(N + 1))  # skip to the right
    nxt_l = array('i', range(N + 1))  # skip to the left
    nxt_d = array('i', range(N + 1))  # skip downward
    nxt_u = array('i', range(N + 1))  # skip upward

    # alive[x] == 1 means wall at cell-id x is still present.
    alive = bytearray(b'\x01') * N

    # DSU‐style find with path‑halving for each direction
    def find_r(x):
        nr = nxt_r
        while nr[x] != x:
            # path halving
            nr[x] = nr[nr[x]]
            x = nr[x]
        return x

    def find_l(x):
        nl = nxt_l
        while nl[x] != x:
            nl[x] = nl[nl[x]]
            x = nl[x]
        return x

    def find_d(x):
        nd = nxt_d
        while nd[x] != x:
            nd[x] = nd[nd[x]]
            x = nd[x]
        return x

    def find_u(x):
        nu = nxt_u
        while nu[x] != x:
            nu[x] = nu[nu[x]]
            x = nu[x]
        return x

    remaining = N

    for _ in range(Q):
        r, c = map(int, input().split())
        r0 = r - 1
        c0 = c - 1
        idx = r0 * W + c0

        if alive[idx]:
            # Destroy this wall only
            alive[idx] = 0
            remaining -= 1
            # Link out idx from the skip‐lists in all 4 directions
            if c0 + 1 < W:
                nxt_r[idx] = idx + 1
            else:
                nxt_r[idx] = sentinel
            if c0 - 1 >= 0:
                nxt_l[idx] = idx - 1
            else:
                nxt_l[idx] = sentinel
            if r0 + 1 < H:
                nxt_d[idx] = idx + W
            else:
                nxt_d[idx] = sentinel
            if r0 - 1 >= 0:
                nxt_u[idx] = idx - W
            else:
                nxt_u[idx] = sentinel
        else:
            # Cell already destroyed: destroy the first wall in each direction, if any
            # UP
            x2 = find_u(idx)
            if x2 != sentinel and alive[x2]:
                alive[x2] = 0
                remaining -= 1
                rr = x2 // W
                cc = x2 % W
                if cc + 1 < W:
                    nxt_r[x2] = x2 + 1
                else:
                    nxt_r[x2] = sentinel
                if cc - 1 >= 0:
                    nxt_l[x2] = x2 - 1
                else:
                    nxt_l[x2] = sentinel
                if rr + 1 < H:
                    nxt_d[x2] = x2 + W
                else:
                    nxt_d[x2] = sentinel
                if rr - 1 >= 0:
                    nxt_u[x2] = x2 - W
                else:
                    nxt_u[x2] = sentinel

            # DOWN
            x2 = find_d(idx)
            if x2 != sentinel and alive[x2]:
                alive[x2] = 0
                remaining -= 1
                rr = x2 // W
                cc = x2 % W
                if cc + 1 < W:
                    nxt_r[x2] = x2 + 1
                else:
                    nxt_r[x2] = sentinel
                if cc - 1 >= 0:
                    nxt_l[x2] = x2 - 1
                else:
                    nxt_l[x2] = sentinel
                if rr + 1 < H:
                    nxt_d[x2] = x2 + W
                else:
                    nxt_d[x2] = sentinel
                if rr - 1 >= 0:
                    nxt_u[x2] = x2 - W
                else:
                    nxt_u[x2] = sentinel

            # LEFT
            x2 = find_l(idx)
            if x2 != sentinel and alive[x2]:
                alive[x2] = 0
                remaining -= 1
                rr = x2 // W
                cc = x2 % W
                if cc + 1 < W:
                    nxt_r[x2] = x2 + 1
                else:
                    nxt_r[x2] = sentinel
                if cc - 1 >= 0:
                    nxt_l[x2] = x2 - 1
                else:
                    nxt_l[x2] = sentinel
                if rr + 1 < H:
                    nxt_d[x2] = x2 + W
                else:
                    nxt_d[x2] = sentinel
                if rr - 1 >= 0:
                    nxt_u[x2] = x2 - W
                else:
                    nxt_u[x2] = sentinel

            # RIGHT
            x2 = find_r(idx)
            if x2 != sentinel and alive[x2]:
                alive[x2] = 0
                remaining -= 1
                rr = x2 // W
                cc = x2 % W
                if cc + 1 < W:
                    nxt_r[x2] = x2 + 1
                else:
                    nxt_r[x2] = sentinel
                if cc - 1 >= 0:
                    nxt_l[x2] = x2 - 1
                else:
                    nxt_l[x2] = sentinel
                if rr + 1 < H:
                    nxt_d[x2] = x2 + W
                else:
                    nxt_d[x2] = sentinel
                if rr - 1 >= 0:
                    nxt_u[x2] = x2 - W
                else:
                    nxt_u[x2] = sentinel

    print(remaining)

if __name__ == "__main__":
    main()