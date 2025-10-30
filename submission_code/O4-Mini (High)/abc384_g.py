import sys
import threading
def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(N)]
    K = int(next(it))
    # Read queries, store as (Y, X, original_index)
    queries = []
    for qi in range(K):
        x = int(next(it))
        y = int(next(it))
        queries.append((y, x, qi))
    # Sort queries by Y (prefix length of B)
    queries.sort()
    # Prefix sums of A and B
    SA = [0] * (N + 1)
    SB = [0] * (N + 1)
    for i in range(N):
        SA[i+1] = SA[i] + A[i]
        SB[i+1] = SB[i] + B[i]
    # Build segment tree over A[0..N-1], size = next power of two
    # We'll store at each node:
    #   minval[node], maxval[node], sumval[node], count[node]
    #   g1[node], g2[node] : accumulated contributions
    n2 = 1 << ((N-1).bit_length())
    size = 2 * n2
    INF = 10**18
    minval = [0] * size
    maxval = [0] * size
    sumval = [0] * size
    countarr = [0] * size
    g1 = [0] * size
    g2 = [0] * size
    # Initialize leaves
    for i in range(n2):
        idx = n2 + i
        if i < N:
            v = A[i]
            minval[idx] = v
            maxval[idx] = v
            sumval[idx] = v
            countarr[idx] = 1
        else:
            # dummy leaves
            minval[idx] = INF
            maxval[idx] = -INF
            sumval[idx] = 0
            countarr[idx] = 0
    # Build internal nodes
    for idx in range(n2-1, 0, -1):
        lc = idx*2
        rc = lc + 1
        mn = minval[lc]
        mv = minval[rc]
        if mv < mn: mn = mv
        mx = maxval[lc]
        mv2 = maxval[rc]
        if mv2 > mx: mx = mv2
        minval[idx] = mn
        maxval[idx] = mx
        sumval[idx] = sumval[lc] + sumval[rc]
        countarr[idx] = countarr[lc] + countarr[rc]
    # Prepare answer array
    ans = [0] * K
    curY = 0
    # Local aliases for speed
    minv = minval; maxv = maxval; sumv = sumval; cnt = countarr
    g1l = g1; g2l = g2
    # Process queries in increasing Y
    for y, x, qi in queries:
        # Process B events up to y (prefix length)
        while curY < y:
            b = B[curY]
            # Update segment tree for threshold b:
            # for all A_i >= b, add A_i and add b
            stack = [1]
            while stack:
                node = stack.pop()
                # If all values in node are < b, skip
                if maxv[node] < b:
                    continue
                # If all values in node are >= b, full coverage
                if minv[node] >= b:
                    g1l[node] += sumv[node]
                    g2l[node] += cnt[node] * b
                else:
                    # Partial: push children
                    # right then left (order doesn't matter)
                    stack.append(node*2+1)
                    stack.append(node*2)
            curY += 1
        # Now answer query D = sum_{i< x} g1[i] - g2[i]
        # That is sum g1/g2 on leaves [0..x-1] => tree indices [n2..n2+x-1]
        l = n2
        r = n2 + x - 1
        res1 = 0
        res2 = 0
        while l <= r:
            if (l & 1):
                res1 += g1l[l]
                res2 += g2l[l]
                l += 1
            if not (r & 1):
                res1 += g1l[r]
                res2 += g2l[r]
                r -= 1
            l //= 2
            r //= 2
        D = res1 - res2
        # F = x * SB[y] - y * SA[x] + 2*D
        ans[qi] = x * SB[y] - y * SA[x] + (D << 1)
    # Output answers in original order
    out = sys.stdout.write
    for v in ans:
        out(str(v))
        out('
')

if __name__ == "__main__":
    main()