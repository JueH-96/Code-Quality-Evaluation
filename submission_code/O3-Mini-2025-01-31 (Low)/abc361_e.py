def main():
    import sys
    from collections import deque
    sys.setrecursionlimit(10**7)
    
    input = sys.stdin.readline
    N = int(input())
    graph = [[] for _ in range(N + 1)]
    total_distance = 0

    for _ in range(N - 1):
        a, b, c = map(int, input().split())
        graph[a].append((b, c))
        graph[b].append((a, c))
        total_distance += c

    def bfs(start):
        """ Performs BFS from the start node and returns the farthest node and its distance. """
        dist = [-1] * (N + 1)
        dq = deque([start])
        dist[start] = 0
        farthest_node = start

        while dq:
            node = dq.popleft()
            for neighbor, weight in graph[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + weight
                    dq.append(neighbor)
                    if dist[neighbor] > dist[farthest_node]:
                        farthest_node = neighbor
        return farthest_node, dist[farthest_node]

    # Compute the diameter of the tree using two BFS steps.
    # 1. Find the farthest node from an arbitrary node (starting with node 1).
    far_node, _ = bfs(1)
    # 2. Find the farthest node from the far_node found, the distance between them is the tree's diameter.
    _, diameter = bfs(far_node)

    # The minimum travel distance is equal to "twice the total distance on the tree minus the diameter".
    # This is because in a tree you normally have to traverse each road twice,
    # but you can save the distance of the longest path if you finish at one end of it.
    answer = 2 * total_distance - diameter
    sys.stdout.write(str(answer))

if __name__ == '__main__':
    main()