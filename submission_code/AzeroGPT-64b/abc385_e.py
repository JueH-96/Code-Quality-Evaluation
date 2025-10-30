from collections import deque
import sys

readlines = sys.stdin.readlines
readline = sys.stdin.readline

class Graph():
    def __init__(self, nodes_num) -> None:
        _ = nodes_num
        self.edges = {}
    
    def add_edge(self, u, v):
        if u not in self.edges:
            self.edges[u] = []
        self.edges[u].append(v)

    def BFS(self, root):
        visited = {root}
        queue = deque([root])

        while queue:
            current = queue.popleft()
            node_path = self.edges.get(current)
            if node_path:
                for child in node_path:
                    if child not in visited:
                        visited.add(child)
                        queue.append(child)
        return visited

    def create_adj_list(self):
        adj_list = {node: set(neighbors) for node, neighbors in self.edges.items()}
        return adj_list
def count_neighbor_degrees(nodes_count, edges):
    nodes = [set() for _ in range(nodes_count + 1)]
    degrees = [0] * (nodes_count + 1)

    for u, v in edges:
        nodes[u].add(v)
        nodes[v].add(u)
        degrees[u] += 1
        degrees[v] += 1
    return degrees, nodes

def check_deletionOrder(nodes_count, degrees, nodes):
    deletionOrder = [0] * nodes_count
    for i in range(1, nodes_count + 1):
        if degrees[i] == 0:
            continue
        if len(nodes[i]) >= 2 and all(degrees[j] == 1 or degrees[j] >= len(nodes[j]) - 1 for j in nodes[i]):
            deletionOrder[degrees[i] - 1] += 1
            for j in nodes[i]:
                degrees[j] -= 1
    return deletionOrder

def optimised_snowflakeTree(nodes_count, edges):
    degrees, nodes = count_neighbor_degrees(nodes_count, edges)
    nodesWithLargeDeg = [i for i, deg in enumerate(degrees[1:], 1) if deg > 1]
    

    if len(nodesWithLargeDeg) == 1 and degrees[nodesWithLargeDeg[0]] == 2:
        return 0
    
    if len(nodesWithLargeDeg) == 2 and degrees[nodesWithLargeDeg[0]] == 2:
        if len(nodes[nodesWithLargeDeg[0]]) == 1 or len(nodes[nodesWithLargeDeg[1]]) == 1:
            return 1
    elif len(nodesWithLargeDeg) == 0 and nodes_count > 2:
        return nodes_count - 2
    
    deletionOrder = check_deletionOrder(nodes_count, degrees, nodes)
    
    if sum(deletionOrder[2:]) == 1 and max(deletionOrder) == 3:
        return nodes_count - 3
    elif sum(deletionOrder[2:]) <= 2:
        return nodes_count - max(deletionOrder) + 1
    else:
        return nodes_count - (max(deletionOrder) - sum(deletionOrder[:2]) + 1)

nodes_count = int(readline())
edges = [tuple(map(int, input().split())) for _ in range(nodes_count - 1)]

answer = optimised_snowflakeTree(nodes_count, edges)
print(answer)