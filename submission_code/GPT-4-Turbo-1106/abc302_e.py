class Graph:
    def __init__(self, N):
        self.N = N
        self.connected = [0] * (N + 1)
        self.isolated_count = N

    def connect(self, u, v):
        if self.connected[u] == 0:
            self.isolated_count -= 1
        if self.connected[v] == 0:
            self.isolated_count -= 1
        self.connected[u] += 1
        self.connected[v] += 1

    def disconnect(self, v):
        if self.connected[v] > 0:
            self.isolated_count += 1
        self.connected[v] = 0

    def get_isolated_count(self):
        return self.isolated_count

def process_queries(N, Q, queries):
    graph = Graph(N)
    for query in queries:
        if query[0] == 1:
            _, u, v = query
            graph.connect(u, v)
        elif query[0] == 2:
            _, v = query
            graph.disconnect(v)
        print(graph.get_isolated_count())

# Read input from stdin
N, Q = map(int, input().split())
queries = [list(map(int, input().split())) for _ in range(Q)]

# Process queries
process_queries(N, Q, queries)