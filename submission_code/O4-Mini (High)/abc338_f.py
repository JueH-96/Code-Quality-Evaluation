import sys
from itertools import combinations

def main():
    input = sys.stdin.readline
    INF = 10**15

    # Read input
    N, M = map(int, input().split())
    dist = [[INF]*N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0
    for _ in range(M):
        u, v, w = map(int, input().split())
        u -= 1; v -= 1
        if w < dist[u][v]:
            dist[u][v] = w

    # Floyd-Warshall for all-pairs shortest paths
    for k in range(N):
        dk = dist[k]
        for i in range(N):
            di = dist[i]
            ik = di[k]
            if ik == INF: 
                continue
            # relax i->j via k
            s = ik
            for j in range(N):
                alt = s + dk[j]
                if alt < di[j]:
                    di[j] = alt

    # Check if there is some start that can reach all nodes
    full_mask = (1 << N) - 1
    reachable_any = False
    for i in range(N):
        ok = True
        for j in range(N):
            if dist[i][j] == INF:
                ok = False
                break
        if ok:
            reachable_any = True
            break
    if not reachable_any:
        print("No")
        return

    # DP over subsets: dp_k[mask][i] = min cost to cover 'mask' and end at i.
    # We only store dp for one size-k at a time in a flat list.

    # Initial: k = 1, masks = [1<<i]
    masks = [1<<i for i in range(N)]
    dp_cur = [INF] * (len(masks) * N)
    for idx, m in enumerate(masks):
        # only one bit set: i = bit index
        i = m.bit_length() - 1
        dp_cur[idx*N + i] = 0

    # Iterate k = 1 .. N-1
    for k in range(1, N):
        # build list of all masks of size k+1
        next_masks = []
        append_nm = next_masks.append
        for comb in combinations(range(N), k+1):
            mb = 0
            for x in comb:
                mb |= 1 << x
            append_nm(mb)
        # map mask -> index in next_masks
        m2idx = {m: i for i, m in enumerate(next_masks)}

        # prepare dp for k+1
        Lnext = len(next_masks)
        dp_next = [INF] * (Lnext * N)

        # locals for speed
        dp_c = dp_cur
        dp_n = dp_next
        DN = N
        D = dist
        FM = full_mask

        # transitions
        for idx, m in enumerate(masks):
            base = idx * DN
            # iterate bits i in m
            mm = m
            while mm:
                b = mm & -mm
                i = b.bit_length() - 1
                cost_i = dp_c[base + i]
                if cost_i < INF:
                    di = D[i]
                    rem = FM ^ m
                    rr = rem
                    # iterate bits j not in m
                    while rr:
                        b2 = rr & -rr
                        j = b2.bit_length() - 1
                        d_ij = di[j]
                        if d_ij < INF:
                            nc = cost_i + d_ij
                            ni = m2idx[m | b2] * DN + j
                            if nc < dp_n[ni]:
                                dp_n[ni] = nc
                        rr ^= b2
                mm ^= b

        # move to next
        masks = next_masks
        dp_cur = dp_next

    # Now k = N, masks = [full_mask], dp_cur holds costs ending at each i
    ans = min(dp_cur[i] for i in range(N))
    print(ans)

if __name__ == "__main__":
    main()