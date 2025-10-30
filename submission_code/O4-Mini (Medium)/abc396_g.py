import sys
import threading

def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    H = int(line[0])
    W = int(line[1])
    N = 1 << W

    # Read the grid rows and build frequency array
    freq = [0] * N
    for _ in range(H):
        s = data.readline().strip()
        m = 0
        for ch in s:
            m = (m << 1) | (ch == '1')
        freq[m] += 1

    # Precompute popcounts for 0..N-1
    pc = [0] * N
    for i in range(1, N):
        pc[i] = pc[i >> 1] + (i & 1)

    # Build G array: cost for a single row mask m (without row flip)
    # G[m] = min(popcount(m), W - popcount(m))
    G = [0] * N
    halfW = W
    for m in range(N):
        cnt = pc[m]
        # if we choose r_x=0, cost is cnt; if r_x=1, cost is W-cnt
        G[m] = cnt if cnt <= halfW - cnt else (halfW - cnt)

    # Fast Walsh-Hadamard Transform for XOR-convolution
    def fwht(a):
        n = len(a)
        length = 1
        while length < n:
            for i in range(0, n, length * 2):
                for j in range(i, i + length):
                    u = a[j]
                    v = a[j + length]
                    a[j] = u + v
                    a[j + length] = u - v
            length <<= 1

    # Copy arrays to transform
    fa = freq[:]  # freq doesn't need after
    fb = G[:]     # G doesn't need after

    # Forward transform
    fwht(fa)
    fwht(fb)
    # Pointwise multiply
    for i in range(N):
        fa[i] = fa[i] * fb[i]
    # Inverse transform (same as forward for XOR)
    fwht(fa)
    # After inverse, we must divide by N
    # fa[c] now holds sum_{m} freq[m] * G[m xor c]
    invN = 1 << W  # N
    best = None
    for c in range(N):
        val = fa[c] // invN
        if best is None or val < best:
            best = val

    print(best)

if __name__ == "__main__":
    main()