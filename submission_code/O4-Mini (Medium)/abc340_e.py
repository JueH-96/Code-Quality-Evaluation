import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(M)]
    # Fenwick for range add, point query
    size = N + 5
    fenw = [0] * size
    def fenw_add(pos, val):
        # pos: 0-based, add val at D[pos]
        i = pos + 1
        while i < size:
            fenw[i] += val
            i += i & -i
    def fenw_sum(pos):
        # sum D[0..pos]
        i = pos + 1
        s = 0
        while i > 0:
            s += fenw[i]
            i -= i & -i
        return s

    # override array
    O = [0] * N
    # Initialize override so that X[i] = A[i]
    # X[i] = add_all - O[i] + rem_sum[i]; initially add_all=0, rem_sum=0 => O[i] = -A[i]
    for i in range(N):
        O[i] = -A[i]

    add_all = 0

    for b in B:
        # compute current X[b]
        curr_rem = fenw_sum(b)
        k = add_all - O[b] + curr_rem
        if k == 0:
            # nothing to do
            continue
        full = k // N
        rem = k - full * N
        add_all += full
        if rem:
            # distribute rem extra 1s to boxes (b+1)%N ... (b+rem)%N
            L = (b + 1) % N
            R = (b + rem) % N
            if L <= R:
                # range L..R
                fenw_add(L, 1)
                fenw_add(R + 1, -1)
            else:
                # wrap: L..N-1 and 0..R
                fenw_add(L, 1)
                fenw_add(N, -1)
                fenw_add(0, 1)
                fenw_add(R + 1, -1)
        # reset box b to full
        # we need O[b] so that: add_all - O[b] + rem_sum[b] == full
        # => O[b] = add_all + rem_sum[b] - full
        curr_rem_b = fenw_sum(b)
        O[b] = add_all + curr_rem_b - full

    # compute final X
    out = []
    for i in range(N):
        val = add_all - O[i] + fenw_sum(i)
        out.append(str(val))
    sys.stdout.write(" ".join(out))

if __name__ == "__main__":
    main()