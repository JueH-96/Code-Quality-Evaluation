import sys
import bisect

MOD = 998244353

def main():
    data = sys.stdin.read().split()
    if not data: 
        print(0)
        return
    it = iter(data)
    n = int(next(it)); m = int(next(it)); k_step = int(next(it))
    edges = []
    for i in range(m):
        u = int(next(it)); v = int(next(it))
        edges.append((u, v))

    special_set = set([1])
    for u, v in edges:
        special_set.add(u)

    S_sorted = sorted(special_set)
    L = len(S_sorted)
    index_map = {}
    for idx, node in enumerate(S_sorted):
        index_map[node] = idx

    gaps = [0] * L
    for i in range(L):
        next_i = (i+1) % L
        gap_val = (S_sorted[next_i] - S_sorted[i] + n) % n
        if gap_val == 0:
            gap_val = n
        gaps[i] = gap_val

    non_special_targets = set()
    for u, v in edges:
        if v not in special_set:
            non_special_targets.add(v)

    next_spec = {}
    dist_spec = {}
    if non_special_targets:
        for t in non_special_targets:
            pos = bisect.bisect_left(S_sorted, t+1)
            if pos < len(S_sorted):
                next_spec[t] = S_sorted[pos]
                dist_spec[t] = next_spec[t] - t
            else:
                next_spec[t] = S_sorted[0]
                dist_spec[t] = next_spec[t] + n - t

    edges_from = {}
    for s in special_set:
        edges_from[s] = []
    for u, v in edges:
        if u in special_set:
            edges_from[u].append(v)

    dp1 = [[0] * L for _ in range(k_step+1)]
    idx1 = index_map[1]
    dp1[0][idx1] = 1

    non_special_count = 0

    for k in range(0, k_step):
        for i in range(L):
            ways = dp1[k][i]
            if ways == 0:
                continue
            s = S_sorted[i]
            d_cycle = gaps[i]
            next_k = k + d_cycle
            if next_k <= k_step:
                j = (i+1) % L
                dp1[next_k][j] = (dp1[next_k][j] + ways) % MOD
            else:
                non_special_count = (non_special_count + ways) % MOD

            if s in edges_from:
                for t in edges_from[s]:
                    if t in special_set:
                        j = index_map[t]
                        next_k = k + 1
                        if next_k <= k_step:
                            dp1[next_k][j] = (dp1[next_k][j] + ways) % MOD
                    else:
                        next_k = k + 1
                        if next_k > k_step:
                            continue
                        if next_k == k_step:
                            non_special_count = (non_special_count + ways) % MOD
                        else:
                            d = dist_spec[t]
                            if next_k + d <= k_step:
                                j = index_map[next_spec[t]]
                                dp1[next_k+d][j] = (dp1[next_k+d][j] + ways) % MOD
                            else:
                                non_special_count = (non_special_count + ways) % MOD

    total = (sum(dp1[k_step]) + non_special_count) % MOD
    print(total)

if __name__ == '__main__':
    main()