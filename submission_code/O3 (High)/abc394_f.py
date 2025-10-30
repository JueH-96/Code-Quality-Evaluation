import sys
sys.setrecursionlimit(1 << 25)

def main() -> None:
    import sys
    input = sys.stdin.readline

    N = int(input())
    adj = [[] for _ in range(N)]
    deg = [0] * N
    for _ in range(N - 1):
        a, b = map(int, input().split())
        a -= 1
        b -= 1
        adj[a].append(b)
        adj[b].append(a)
        deg[a] += 1
        deg[b] += 1

    # vertices whose original degree is at least 4 can become internal (degree 4) nodes
    is_high = [d >= 4 for d in deg]

    visited = [False] * N
    max_comp = 0

    from collections import deque

    for v in range(N):
        if is_high[v] and not visited[v]:
            q = deque([v])
            visited[v] = True
            sz = 0
            while q:
                u = q.popleft()
                sz += 1
                for w in adj[u]:
                    if is_high[w] and not visited[w]:
                        visited[w] = True
                        q.append(w)
            max_comp = max(max_comp, sz)

    # no vertex can be internal
    if max_comp == 0:
        print(-1)
        return

    # at most (N-2)//3 internal vertices are possible
    upper = (N - 2) // 3
    best_internal = min(max_comp, upper)

    if best_internal == 0:          # N < 5
        print(-1)
        return

    # total vertices in the largest alkane
    answer = 3 * best_internal + 2
    print(answer)

if __name__ == "__main__":
    main()