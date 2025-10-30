import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    M = int(data[0])
    S = [data[1], data[2], data[3]]
    INF = 10**18
    answer = INF
    # Try each digit d
    for d in '0123456789':
        # positions on each reel where digit d appears
        P = []
        valid = True
        for i in range(3):
            pts = [idx for idx, ch in enumerate(S[i]) if ch == d]
            if not pts:
                valid = False
                break
            P.append(pts)
        if not valid:
            continue
        # For each reel, build the set of possible stopping times
        # using k = 0,1,2 (since 3 reels => up to 3 distinct multiples needed)
        T_i = []
        for i in range(3):
            times = []
            for r in P[i]:
                for k in range(3):
                    times.append(r + k * M)
            T_i.append(sorted(set(times)))
        # All candidate times
        all_times = sorted(set(T_i[0] + T_i[1] + T_i[2]))
        # Map time -> index in all_times
        t2idx = {t: idx for idx, t in enumerate(all_times)}
        R = len(all_times)
        # Build adjacency: reel i can use right-node index v if all_times[v] is in T_i[i]
        adj = []
        for i in range(3):
            lst = [t2idx[t] for t in T_i[i]]
            lst.sort()
            adj.append(lst)
        # Bipartite matching check: can we match all 3 reels to distinct times <= all_times[idx_end] ?
        def can_match(idx_end):
            matchR = [-1] * R
            def dfs(u):
                for v in adj[u]:
                    if v > idx_end:
                        break
                    if not seen[v]:
                        seen[v] = True
                        if matchR[v] == -1 or dfs(matchR[v]):
                            matchR[v] = u
                            return True
                return False
            # Try to match each reel
            for u in range(3):
                seen = [False] * (idx_end + 1)
                if not dfs(u):
                    return False
            return True

        # Binary search on all_times to find minimal T for which matching is possible
        l, r = 0, R - 1
        best_idx = None
        while l <= r:
            mid = (l + r) // 2
            if can_match(mid):
                best_idx = mid
                r = mid - 1
            else:
                l = mid + 1
        if best_idx is not None:
            answer = min(answer, all_times[best_idx])

    # Output result
    if answer == INF:
        print(-1)
    else:
        print(answer)

if __name__ == "__main__":
    main()