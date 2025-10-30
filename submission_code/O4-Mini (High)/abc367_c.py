def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    R = [int(next(it)) for _ in range(N)]
    out = []
    seq = [0] * N

    def dfs(pos, s):
        if pos == N:
            if s % K == 0:
                out.append(' '.join(map(str, seq)))
            return
        # iterate values for position pos
        ri = R[pos]
        for v in range(1, ri + 1):
            seq[pos] = v
            dfs(pos + 1, s + v)

    dfs(0, 0)
    sys.stdout.write('
'.join(out))

if __name__ == "__main__":
    main()