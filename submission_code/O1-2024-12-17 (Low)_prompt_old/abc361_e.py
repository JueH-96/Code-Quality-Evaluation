def solve():
    import sys
    sys.setrecursionlimit(10**7)
    from collections import deque
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    edges_data = data[1:]

    # Adjacency list for the tree
    adj = [[] for _ in range(N + 1)]
    
    # Read edges, build graph, track sum of all edge lengths
    total_length = 0
    idx = 0
    for _ in range(N - 1):
        A = int(edges_data[idx])
        B = int(edges_data[idx + 1])
        C = int(edges_data[idx + 2])
        idx += 3
        adj[A].append((B, C))
        adj[B].append((A, C))
        total_length += C

    # BFS function to find the farthest node and distance starting from 'start'
    def bfs(start):
        dist = [-1] * (N + 1)
        dist[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for nx, w in adj[v]:
                if dist[nx] < 0:
                    dist[nx] = dist[v] + w
                    queue.append(nx)
        # Find the farthest node from 'start' and its distance
        farthest_node = 1
        max_dist = -1
        for i in range(1, N + 1):
            if dist[i] > max_dist:
                max_dist = dist[i]
                farthest_node = i
        return farthest_node, max_dist

    # 1) Choose an arbitrary node (e.g., 1) and find the farthest node X from it.
    X, _ = bfs(1)
    # 2) From X, find the farthest node Y and the distance between X and Y (which is the diameter).
    Y, diameter = bfs(X)
    
    # The minimum route = 2 * (sum of all edges) - diameter
    result = 2 * total_length - diameter
    
    print(result)

def main():
    solve()

if __name__ == "__main__":
    main()