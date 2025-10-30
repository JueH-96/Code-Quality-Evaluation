from collections import defaultdict, deque
import math

class Graph:
    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.adj_list = defaultdict(dict)

    def add_edge(self, u, v, t, index):
        self.adj_list[u][v] = {'time': t, 'index': index}
        self.adj_list[v][u] = {'time': t, 'index': index}

def shortest_path_with_must_pass(N, M, paths, queries):
    graph = Graph(N, M)
    for u, v, t in paths:
        graph.add_edge(u, v, t)

    answers = []
    for K, required_bridges in queries:
        # Convert to a set for O(1) lookups
        required_bridges_set = set(required_bridges)
        visited_bridges = set()
        queue = deque([(1, 0, frozenset())])
        visited = {1: (0, frozenset(()))}  # (prev_cost, prev_bridges)
        reached = False
        min_cost = math.inf

        while queue:
            current, cost, used_bridges = queue.popleft()

            # Check if all required bridges have been used
            if required_bridges_set.issubset(used_bridges):
                if current == N:
                    reached = True
                    min_cost = min(min_cost, cost)
                    break
                continue  # Don't process further if N is reached and all bridges are used

            for neighbor, data in graph.adj_list[current].items():
                if data['index'] not in used_bridges:
                    new_cost = cost + data['time']
                    new_used_bridges = frozenset(list(used_bridges) + [data['index']])

                    if neighbor not in visited or visited[neighbor][0] > new_cost or visited[neighbor][1] < new_used_bridges:
                        visited[neighbor] = (new_cost, new_used_bridges)
                        queue.append((neighbor, new_cost, new_used_bridges))

        answers.append(min_cost if reached else -1)

    print('
'.join(map(str, answers)))

# Reading the inputs from stdin as per the format provided
N, M = map(int, input().split())
paths = [tuple(map(int, input().split())) for _ in range(M)]
Q = int(input())
queries = [(int(input()), [int(i) for i in input().split()]) for _ in range(Q)]
shortest_path_with_must_pass(N, M, paths, queries)