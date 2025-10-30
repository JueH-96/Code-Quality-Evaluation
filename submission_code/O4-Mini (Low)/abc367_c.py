def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().split()
    N = int(data[0])
    K = int(data[1])
    R = list(map(int, data[2:]))

    current = [0] * N
    out_lines = []

    def dfs(pos, s):
        if pos == N:
            if s % K == 0:
                # Record valid sequence
                out_lines.append(" ".join(map(str, current)))
            return
        # Try all values from 1 to R[pos]
        for v in range(1, R[pos] + 1):
            current[pos] = v
            dfs(pos + 1, s + v)

    dfs(0, 0)
    # Print all found sequences; if none, prints nothing
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()