import copy
from collections import deque
from typing import List

class Edge:
    def __init__(self, to, rev, capacity, cost):
        self.to = to
        self.rev = rev
        self.capacity = capacity
        self.cost = cost

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        if rows == 0:
            return 0
        cols = len(grid[0]) if rows > 0 else 0

        # Collect unique numbers and their occurrences per row
        unique_numbers = set()
        row_numbers = []
        for row in grid:
            current = set()
            for num in row:
                current.add(num)
                unique_numbers.add(num)
            row_numbers.append(list(current))
        unique_numbers = list(unique_numbers)
        num_count = len(unique_numbers)

        # Assign node IDs
        source = 0
        sink = rows + num_count + 1
        node_count = sink + 1

        # Build the original graph
        original_graph = [[] for _ in range(node_count)]
        number_to_node = {num: i for i, num in enumerate(unique_numbers)}

        # Add edges from source to row nodes
        for row in range(rows):
            row_node = row + 1
            rev = len(original_graph[row_node])
            original_graph[source].append(Edge(row_node, rev, 1, 0))
            rev_source = len(original_graph[source]) - 1
            original_graph[row_node].append(Edge(source, rev_source, 0, 0))

        # Add edges from row nodes to number nodes
        for row in range(rows):
            row_node = row + 1
            for num in row_numbers[row]:
                num_node = rows + 1 + number_to_node[num]
                rev = len(original_graph[num_node])
                original_graph[row_node].append(Edge(num_node, rev, 1, -num))
                rev_row = len(original_graph[row_node]) - 1
                original_graph[num_node].append(Edge(row_node, rev_row, 0, num))

        # Add edges from number nodes to sink
        for num in unique_numbers:
            num_node = rows + 1 + number_to_node[num]
            rev = len(original_graph[sink])
            original_graph[num_node].append(Edge(sink, rev, 1, 0))
            rev_num = len(original_graph[num_node]) - 1
            original_graph[sink].append(Edge(num_node, rev_num, 0, 0))

        max_sum = 0

        # Try all possible flow sizes from 1 to rows
        for k in range(1, rows + 1):
            # Create a deep copy of the graph for each k
            graph = copy.deepcopy(original_graph)
            cost = self.min_cost_flow(graph, source, sink, k)
            if cost is not None:
                current_sum = -cost
                if current_sum > max_sum:
                    max_sum = current_sum

        return max_sum

    def min_cost_flow(self, graph, source, sink, required_flow):
        res = 0
        cost = 0
        while res < required_flow:
            # Use Bellman-Ford to find the shortest path
            dist = [float('inf')] * len(graph)
            dist[source] = 0
            prev = [-1] * len(graph)
            inqueue = [False] * len(graph)
            queue = deque()
            queue.append(source)
            inqueue[source] = True

            while queue:
                u = queue.popleft()
                inqueue[u] = False
                for i, e in enumerate(graph[u]):
                    if e.capacity > 0 and dist[e.to] > dist[u] + e.cost:
                        dist[e.to] = dist[u] + e.cost
                        prev[e.to] = (u, i)
                        if not inqueue[e.to]:
                            queue.append(e.to)
                            inqueue[e.to] = True

            if dist[sink] == float('inf'):
                return None  # No more augmenting paths

            # Find the minimum capacity along the path
            flow = required_flow - res
            v = sink
            path = []
            while v != source:
                u, i = prev[v]
                path.append((u, i))
                v = u

            min_cap = flow
            for (u, i) in path:
                e = graph[u][i]
                if e.capacity < min_cap:
                    min_cap = e.capacity

            if min_cap <= 0:
                return None

            # Augment along the path
            v = sink
            while v != source:
                u, i = prev[v]
                e = graph[u][i]
                e.capacity -= min_cap
                rev_e = graph[e.to][e.rev]
                rev_e.capacity += min_cap
                cost += min_cap * e.cost
                v = u

            res += min_cap

        return cost