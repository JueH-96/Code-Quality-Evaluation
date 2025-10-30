import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10000)
    input = sys.stdin.readline

    N = int(input())
    P = [tuple(map(int, input().split())) for _ in range(N)]
    Q = [tuple(map(int, input().split())) for _ in range(N)]

    # cross product of (bx - ax, by - ay) x (cx - ax, cy - ay)
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    # Recursively find non-crossing perfect matching
    # P_idxs, Q_idxs are lists of indices into P and Q
    def solve(P_idxs, Q_idxs):
        n = len(P_idxs)
        if n == 0:
            return []
        # pick first P
        pi = P_idxs[0]
        p = P[pi]
        # try match with each candidate Q
        for qj in Q_idxs:
            q = Q[qj]
            leftP = []
            rightP = []
            # classify other Ps
            for idx in P_idxs[1:]:
                c = cross(p, q, P[idx])
                if c > 0:
                    leftP.append(idx)
                else:
                    rightP.append(idx)
            leftQ = []
            rightQ = []
            # classify other Qs
            for idx in Q_idxs:
                if idx == qj:
                    continue
                c = cross(p, q, Q[idx])
                if c > 0:
                    leftQ.append(idx)
                else:
                    rightQ.append(idx)
            # need equal counts on one side
            if len(leftP) == len(leftQ):
                # good partition
                res = [(pi, qj)]
                res += solve(leftP, leftQ)
                res += solve(rightP, rightQ)
                return res
        # if no matching possible
        return None

    P_idxs = list(range(N))
    Q_idxs = list(range(N))
    matching = solve(P_idxs, Q_idxs)
    if matching is None:
        print(-1)
    else:
        # build R: for each P_i (0-based), the matched Q index+1
        R = [0]*N
        for pi, qj in matching:
            R[pi] = qj+1
        print(" ".join(map(str, R)))

if __name__ == "__main__":
    main()