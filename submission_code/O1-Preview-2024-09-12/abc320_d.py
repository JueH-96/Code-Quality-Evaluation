# YOUR CODE HERE
def main():
    import sys
    import threading

    def solve():
        import sys
        sys.setrecursionlimit(1 << 25)
        N, M = map(int, sys.stdin.readline().split())
        adj = [[] for _ in range(N + 1)]  # 1-indexed

        for _ in range(M):
            Ai, Bi, Xi, Yi = map(int, sys.stdin.readline().split())
            adj[Ai].append((Bi, Xi, Yi))

        positions = [None] * (N + 1)  # positions[0] unused, positions[1..N]
        positions[1] = (0, 0)
        from collections import deque
        queue = deque()
        queue.append(1)

        while queue:
            u = queue.popleft()
            x_u, y_u = positions[u]
            for v, dx, dy in adj[u]:
                if positions[v] is None:
                    positions[v] = (x_u + dx, y_u + dy)
                    queue.append(v)
                else:
                    # Input is consistent, so we can ignore consistency check
                    pass

        for i in range(1, N + 1):
            if positions[i] is None:
                print('undecidable')
            else:
                print('{} {}'.format(positions[i][0], positions[i][1]))

    threading.Thread(target=solve).start()