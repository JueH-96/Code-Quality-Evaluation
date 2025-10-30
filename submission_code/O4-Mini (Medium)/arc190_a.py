import sys
import threading
def main():
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N, M = map(int, input().split())
    segs = []
    for i in range(M):
        l, r = map(int, input().split())
        segs.append((l, r, i))

    # 1) Build bestRAt[L]
    bestRAt = [ -1 ] * (N+2)   # bestRAt[L] = max right seen among intervals starting exactly at L
    bestIdAt = [ -1 ] * (N+2)
    for (l, r, idx) in segs:
        if r > bestRAt[l]:
            bestRAt[l] = r
            bestIdAt[l] = idx

    # 2) Build prefix-farthest "far[s]" and bestIdx[s]
    far = [0]*(N+2)
    bestIdx = [-1]*(N+2)
    curMax = 0
    curId  = -1
    for s in range(1, N+1):
        if bestRAt[s] > curMax:
            curMax = bestRAt[s]
            curId  = bestIdAt[s]
        far[s]    = curMax
        bestIdx[s]= curId

    INF = 10**9

    # A) Pure type-1 cover [1..N]
    def greedy_cover(interval_left, interval_right):
        """ Greedily cover [interval_left..interval_right] with far[],
            return (count, list_of_indices) or (INF,[] ) if impossible. """
        res = []
        cur = interval_left
        while cur <= interval_right:
            if cur<1 or cur> N or far[cur] < cur:
                return INF, []
            idx = bestIdx[cur]
            # that interval covers [someLeft..far[cur]]; we pick it
            res.append(idx)
            cur = far[cur] + 1
        return len(res), res

    cost1, cover1 = greedy_cover(1, N)
    if cost1 >= INF:
        cost1 = INF

    # B) Pure type-2 with two disjoint intervals?
    #    Sort by left, track minR so far.
    cost2 = INF
    pair2 = None
    sorted_by_L = sorted(segs, key=lambda x: x[0])
    minR = 10**18
    minId= -1
    for (l, r, idx) in sorted_by_L:
        if l > minR:
            cost2 = 2
            pair2 = (minId, idx)
            break
        if r < minR:
            minR = r
            minId = idx

    # C) One type-2 + greedy type-1 on the hole [L_i..R_i].
    costComb = INF
    bestI = -1
    bestCoverHole = None

    # We only bother if neither A nor B gave <=2
    bestSoFar = min(cost1, cost2)
    if bestSoFar > 2:
        # Try each i, stop early if we get costComb==2
        for (l, r, idx) in segs:
            # greedy cover [l..r]
            cnt = 0
            cur = l
            ok = True
            temp = []
            while cur <= r and cnt+1 < bestSoFar:
                if far[cur] < cur:
                    ok = False
                    break
                i2 = bestIdx[cur]
                temp.append(i2)
                cnt += 1
                cur = far[cur] + 1
            if not ok or cur <= r:
                continue
            # total cost = 1(type‑2) + cnt(type‑1)
            c = 1 + cnt
            if c < costComb:
                costComb = c
                bestI = idx
                bestCoverHole = temp[:]
                bestSoFar = c
                if costComb == 2:
                    break

    # Pick the best among cost1,cost2,costComb
    best = cost1
    method = '1'
    if cost2 < best:
        best = cost2
        method = '2'
    if costComb < best:
        best = costComb
        method = 'C'

    if best >= INF:
        print(-1)
        return

    # Reconstruct the op[] array.
    ops = [0]*M

    if method == '1':
        # all of cover1[] get op=1
        for idx in cover1:
            ops[idx] = 1

    elif method == '2':
        # the two pair2[] get op=2
        i1, i2 = pair2
        ops[i1] = 2
        ops[i2] = 2

    else:
        # Combination: op bestI = 2, and the ones in bestCoverHole get op=1
        ops[bestI] = 2
        for idx in bestCoverHole:
            if idx == bestI:
                # (Very rare conflict; skip if it occurs.)
                continue
            ops[idx] = 1

    # Output
    print(best)
    print(" ".join(str(x) for x in ops))


if __name__ == "__main__":
    main()