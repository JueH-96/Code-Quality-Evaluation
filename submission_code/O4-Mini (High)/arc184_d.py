import sys
import threading
def main():
    import sys
    input = sys.stdin.readline
    MOD = 998244353

    N = int(input().strip())
    # Read balls and build P: P[x-1] = y
    P = [0] * N
    for _ in range(N):
        x, y = map(int, input().split())
        P[x-1] = y

    # Segment the permutation by comparability connectivity:
    # split at i if min(P[0..i]) > max(P[i+1..N-1])
    prefix_min = [0]*N
    suffix_max = [0]*N
    prefix_min[0] = P[0]
    for i in range(1, N):
        v = P[i]
        pm = prefix_min[i-1]
        prefix_min[i] = v if v < pm else pm
    suffix_max[N-1] = P[N-1]
    for i in range(N-2, -1, -1):
        v = P[i]
        sm = suffix_max[i+1]
        suffix_max[i] = v if v > sm else sm

    segments = []
    last = 0
    # splits at i where prefix_min[i] > suffix_max[i+1]
    for i in range(N-1):
        if prefix_min[i] > suffix_max[i+1]:
            segments.append((last, i))
            last = i+1
    segments.append((last, N-1))

    ans = 1
    # Process each independent segment
    for (l, r) in segments:
        m = r - l + 1
        # extract and compress the values in P[l..r]
        vals = P[l:r+1]
        sorted_vals = sorted(vals)
        # map original values to 0..m-1
        rank = {v:i for i,v in enumerate(sorted_vals)}
        Q = [rank[v] for v in vals]   # Q[i] in [0..m-1]
        # build inverse R: R[y] = position i (0..m-1) with Q[i] = y
        R = [0]*m
        for i, y in enumerate(Q):
            R[y] = i

        # Precompute neighbor masks in the "comparability" graph G:
        # neighbor_mask[y] has bit k set if y!=k and
        #    (y-k)*(R[y]-R[k]) > 0  (i.e. comparable)
        neighbor_mask = [0]*m
        for y in range(m):
            Ry = R[y]
            mask = 0
            # loop k from 0..m-1
            # bit k corresponds to vertex y'=k
            for k in range(m):
                if k == y:
                    continue
                # comparability test:
                # if (y-k)*(Ry-R[k]) > 0 then edge
                if (y - k) * (Ry - R[k]) > 0:
                    mask |= (1 << k)
            neighbor_mask[y] = mask

        # DP over y from m-1 down to 0:
        # dp holds all distinct neighbor-union sets U reachable
        # by independent sets T of processed y's.
        dp = {0}
        for y in range(m-1, -1, -1):
            bit_y = 1 << y
            nmask = neighbor_mask[y]
            # iterate over a snapshot of dp
            # so we can add new states into dp safely
            cur = list(dp)
            for u in cur:
                # can pick y into T only if y not already a neighbor
                if not (u & bit_y):
                    dp.add(u | nmask)
        # number of distinct neighbor-union sets = size of dp
        # survivors sets = complements of these unions, same count
        ans = ans * len(dp) % MOD

    print(ans)

if __name__ == "__main__":
    main()