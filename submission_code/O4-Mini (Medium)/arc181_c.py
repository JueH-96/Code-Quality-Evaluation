def main():
    import sys
    input = sys.stdin.readline
    N = int(input())
    P = list(map(int, input().split()))
    Q = list(map(int, input().split()))
    # Compute inverse permutations (0-based):
    ppos = [0]*N
    qpos = [0]*N
    for i, x in enumerate(P):
        ppos[x-1] = i
    for j, x in enumerate(Q):
        qpos[x-1] = j
    # We will build A so that after we reorder rows by P and columns by Q,
    # the permuted matrix A' has A'[i][j] = '1' iff i+j > N (1-based).
    # In 0-based indices that becomes: ppos[i] + qpos[j] >= N-1
    out = []
    thresh = N - 1
    for i in range(N):
        pi = ppos[i]
        row = []
        # build each character
        for j in range(N):
            if pi + qpos[j] >= thresh:
                row.append('1')
            else:
                row.append('0')
        out.append(''.join(row))
    sys.stdout.write('
'.join(out))

if __name__ == "__main__":
    main()