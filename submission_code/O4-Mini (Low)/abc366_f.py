import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    N, K = map(int, input().split())
    funcs = [tuple(map(int, input().split())) for _ in range(N)]
    # 1) Remove pointwise‐dominated functions
    # Sort by A asc, then by B asc
    funcs.sort()
    nd = []
    maxB = -1
    # We go from largest A downwards so if we see a function with B<=maxB
    # it is dominated by some higher‐A with at least as large B.
    for A, B in reversed(funcs):
        if B > maxB:
            nd.append((A, B))
            maxB = B
    nd.reverse()  # now in ascending A
    # nd is nondominated
    M = len(nd)
    # 2) If still too many, truncate by the top M0 by immediate value at x=1
    M0 = 1000
    if M > M0:
        nd.sort(key=lambda ab: ab[0]+ab[1], reverse=True)
        nd = nd[:M0]
        M = M0
    # Pre‐index them
    As = [a for a,b in nd]
    Bs = [b for a,b in nd]
    # 3) Beam‐search
    # Each state is (current_value, used_mask)
    # But used_mask as a frozenset of indices up to M
    from heapq import nlargest
    beam = [(1, frozenset())]  # start with x=1, used nothing
    for depth in range(K):
        cand = []
        # For each partial state, try extending by each unused f_i
        for x, used in beam:
            # if used size = depth
            for i in range(M):
                if i not in used:
                    nx = As[i]*x + Bs[i]
                    cand.append((nx, used|{i}))
        # pick top W
        W = 200
        if len(cand) > W:
            beam = nlargest(W, cand, key=lambda st: st[0])
        else:
            beam = cand
    # The answer is the max value in the final beam
    ans = 0
    for x, _ in beam:
        if x > ans:
            ans = x
    print(ans)

if __name__ == "__main__":
    main()