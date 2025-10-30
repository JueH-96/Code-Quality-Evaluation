import sys
from collections import deque

def main():
    import sys
    from collections import deque

    N_M_X1 = sys.stdin.readline().split()
    N = int(N_M_X1[0])
    M = int(N_M_X1[1])
    X1 = int(N_M_X1[2])

    # Read trains
    trains = []
    for _ in range(M):
        A, B, S, T = map(int, sys.stdin.readline().split())
        trains.append((A, B, S, T))

    # Build graph
    graph = [[] for _ in range(M + 1)]  # 0 is source
    for i in range(M):
        A_i, B_i, S_i, T_i = trains[i]
        for j in range(M):
            if i != j:
                A_j, B_j, S_j, T_j = trains[j]
                if B_i == A_j and T_i <= S_j:
                    # Add edge from i to j with weight T_i - S_j
                    graph[i + 1].append((j + 1, T_i - S_j))

    # Add edges from source to all trains with weight 0
    for j in range(1, M + 1):
        graph[0].append((j, 0))

    # SPFA algorithm
    def spfa():
        dist = [float('inf')] * (M + 1)
        in_queue = [False] * (M + 1)
        queue = deque()

        # Initialize distances
        dist[1] = X1
        dist[0] = 0

        # Start from source
        queue.append(0)
        in_queue[0] = True

        while queue:
            u = queue.popleft()
            in_queue[u] = False
            for v, w in graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True

        # Set X_j = max(0, dist[j])
        delays = [max(0, dist[j]) for j in range(1, M + 1)]
        return delays[1:]  # Exclude X1

    delays = spfa()
    print(' '.join(map(str, delays)))

if __name__ == "__main__":
    main()