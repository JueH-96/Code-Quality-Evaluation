def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    P = list(map(int, data[1:1+n]))
    Q = list(map(int, data[1+n:1+2*n]))
    # R[k] = original row index of the k-th in P-order (0-indexed)
    R = [p-1 for p in P]
    # C[l] = original col index of the l-th in Q-order (0-indexed)
    C = [q-1 for q in Q]
    # Prepare an n x n grid of '0's
    A = [['0'] * n for _ in range(n)]
    # We build a matrix B in sorted row/col order where
    # B[k][l] = '1' iff k + l >= n (0-indexed).
    # Then we map B[k][l] back: A[R[k]][C[l]] = B[k][l].
    for k in range(n):
        for l in range(n):
            if k + l >= n:
                A[R[k]][C[l]] = '1'
    out = sys.stdout
    for row in A:
        out.write(''.join(row) + '
')

if __name__ == "__main__":
    main()