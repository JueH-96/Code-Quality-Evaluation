import sys
import bisect

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    graph = [[] for _ in range(n+1)]
    deg = [0] * (n+1)
    index = 1
    for _ in range(n-1):
        u = int(data[index])
        v = int(data[index+1])
        index += 2
        graph[u].append(v)
        graph[v].append(u)
        deg[u] += 1
        deg[v] += 1

    ans = n
    for c in range(1, n+1):
        neighbors = graph[c]
        m = len(neighbors)
        if m == 0:
            kept = 1
            deletion = n - kept
            if deletion < ans:
                ans = deletion
            continue

        d_list = [deg[u] - 1 for u in neighbors]
        d_list.sort()
        distinct_vals = set(d_list)
        candidate_max = 0
        for v in distinct_vals:
            if v < 1:
                continue
            idx = bisect.bisect_left(d_list, v)
            count = m - idx
            candidate = (1 + v) * count
            if candidate > candidate_max:
                candidate_max = candidate

        kept = 1 + candidate_max
        deletion = n - kept
        if deletion < ans:
            ans = deletion

    print(ans)

if __name__ == '__main__':
    main()