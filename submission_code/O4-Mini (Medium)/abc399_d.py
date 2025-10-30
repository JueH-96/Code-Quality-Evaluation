import sys
import threading

def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    T = int(next(it))
    out_lines = []
    for _ in range(T):
        n = int(next(it))
        m = 2 * n
        A = [0] * m
        for i in range(m):
            A[i] = int(next(it))
        # Record first/second occurrence positions
        occ = [0] * (n + 1)
        types = [0] * m  # 1 for first occ, 2 for second
        # We don't actually need to store p1/p2 arrays
        for i, x in enumerate(A):
            occ[x] += 1
            if occ[x] == 1:
                types[i] = 1
            else:
                types[i] = 2
        # Build adjacency sets for starts and ends
        S = set()
        E = set()
        # scan adjacent positions
        for i in range(m - 1):
            t1 = types[i]
            t2 = types[i+1]
            if t1 == 1 and t2 == 1:
                # both are first occurrences
                S.add((A[i], A[i+1]))
            elif t1 == 2 and t2 == 2:
                # both are second occurrences
                E.add((A[i], A[i+1]))
        # Count intersection
        # iterate the smaller set
        if len(S) < len(E):
            cnt = sum(1 for pair in S if pair in E)
        else:
            cnt = sum(1 for pair in E if pair in S)
        out_lines.append(str(cnt))
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()