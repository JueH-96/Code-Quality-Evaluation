def main():
    import sys
    from collections import deque

    data = sys.stdin.read().split()
    if not data:
        return

    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    # Read A and B sequences.
    A = [int(next(it)) - 1 for _ in range(M)]
    B = [int(next(it)) - 1 for _ in range(M)]

    # Build graph: vertices from 0 to N-1; each constraint add undirected edge.
    # If a constraint requires the same index to have two different bits, it's impossible.
    graph = [[] for _ in range(N)]
    for u, v in zip(A, B):
        # A self-loop constraint (u equals v) is unsatisfiable.
        if u == v:
            sys.stdout.write("No")
            return

        graph[u].append(v)
        graph[v].append(u)

    # Colors: -1 means uncolored, 0 and 1 are two colors.
    color = [-1] * N

    # Use BFS to color the graph in each connected component.
    for i in range(N):
        if color[i] == -1:
            color[i] = 0
            q = deque([i])
            while q:
                cur = q.popleft()
                for nxt in graph[cur]:
                    if color[nxt] == -1:
                        color[nxt] = 1 - color[cur]
                        q.append(nxt)
                    elif color[nxt] == color[cur]:
                        sys.stdout.write("No")
                        return

    sys.stdout.write("Yes")


if __name__ == '__main__':
    main()