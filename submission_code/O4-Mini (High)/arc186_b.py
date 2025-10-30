import sys
import threading

def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it))
    A = [0] * (N + 1)
    for i in range(1, N + 1):
        A[i] = int(next(it))
    mod = 998244353

    # Compute NX[i] = smallest j>i with A[j] < i, or N+1 if none
    NX = [N+1] * (N + 2)
    # DSU "nextUn" to skip assigned i's
    nextUn = list(range(N+2))
    def find(x):
        # find next unassigned index >= x
        # path-compression
        while nextUn[x] != x:
            nextUn[x] = nextUn[nextUn[x]]
            x = nextUn[x]
        return x

    for j in range(1, N+1):
        # for all i in (A[j], j), set NX[i] = j if not set
        L = A[j] + 1
        i = find(L)
        while i < j:
            NX[i] = j
            # mark i processed
            nextUn[i] = i + 1
            i = find(i)

    # Build intervals [l[i], r[i]]
    l = [0] * (N + 1)
    r = [0] * (N + 1)
    for i in range(1, N+1):
        l[i] = A[i] + 1
        # NX[i] is first j>i with A[j]<i; segment ends at NX[i]-1
        r[i] = NX[i] - 1

    # Sort intervals by l ascending, r descending
    ord_idx = list(range(1, N+1))
    ord_idx.sort(key = lambda i: (l[i], -r[i]))

    # Build segmentation tree as interval containment tree
    children = [[] for _ in range(N+1)]
    stack = []
    # dummy root 0 with interval covering [1,N]
    for i in ord_idx:
        # pop until stack.top interval contains this
        while stack and r[stack[-1]] < r[i]:
            stack.pop()
        if stack:
            p = stack[-1]
        else:
            p = 0
        children[p].append(i)
        stack.append(i)

    # The root of segmentation tree is sole child of dummy root 0
    # Input guaranteed valid, so children[0] has exactly one element
    if not children[0]:
        # no intervals? means N=0? But N>=1 by constraints
        print(1)
        return
    root = children[0][0]

    # Prepare left and right child pointers for binary segmentation tree
    left_child = [0] * (N + 1)
    right_child = [0] * (N + 1)
    for p in range(1, N+1):
        if not children[p]:
            continue
        # children[p] is list of direct segmentation children (<=2)
        # assign left/right by index relative to parent
        for c in children[p]:
            if c < p:
                left_child[p] = c
            else:
                right_child[p] = c
        # valid input ensures at most one left and one right

    # Precompute factorials and inverse factorials
    M = mod
    fac = [1] * (N+1)
    for i in range(1, N+1):
        fac[i] = fac[i-1] * i % M
    invfac = [1] * (N+1)
    invfac[N] = pow(fac[N], M-2, M)
    for i in range(N, 0, -1):
        invfac[i-1] = invfac[i] * i % M

    # DP on the tree in post-order (iterative)
    dp = [0] * (N + 1)
    size = [0] * (N + 1)

    # stack for DFS: (node, state) state=0 enter, state=1 exit
    stack2 = [(root, 0)]
    while stack2:
        node, st = stack2.pop()
        if st == 0:
            # push exit state
            stack2.append((node, 1))
            # push children for enter state
            rc = right_child[node]
            if rc:
                stack2.append((rc, 0))
            lc = left_child[node]
            if lc:
                stack2.append((lc, 0))
        else:
            # compute dp[node]
            lc = left_child[node]
            rc = right_child[node]
            szL = size[lc] if lc else 0
            szR = size[rc] if rc else 0
            # size of this subtree (# nodes)
            size[node] = 1 + szL + szR
            # ways = dp[L] * dp[R] * C(szL+szR, szL)
            ways = 1
            if lc:
                ways = dp[lc]
            if rc:
                ways = ways * dp[rc] % M
            # multiply by binomial
            # choose positions for left subtree among children positions
            ways = ways * fac[szL + szR] % M
            ways = ways * invfac[szL] % M
            ways = ways * invfac[szR] % M
            dp[node] = ways

    # result at root
    print(dp[root] % M)

if __name__ == "__main__":
    main()