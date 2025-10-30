import heapq

def solve():
    n, q = map(int, input().split())
    adj = [[] for _ in range(n)]

    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            u, v = query[1], query[2]
            u -= 1
            v -= 1
            adj[u].append(v)
            adj[v].append(u)
        else:
            v, k = query[1], query[2]
            v -= 1
            connected_component_nodes = []
            visited = [False] * n
            stack = [v]
            visited[v] = True
            connected_component_nodes.append(v + 1)
            while stack:
                u = stack.pop()
                for neighbor in adj[u]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        connected_component_nodes.append(neighbor + 1)
                        stack.append(neighbor)

            if len(connected_component_nodes) < k:
                print("-1")
            else:
                min_heap = []
                for node_val in connected_component_nodes:
                    if len(min_heap) < k:
                        heapq.heappush(min_heap, node_val)
                    elif node_val > min_heap[0]:
                        heapq.heappop(min_heap)
                        heapq.heappush(min_heap, node_val)
                print(min_heap[0])

solve()