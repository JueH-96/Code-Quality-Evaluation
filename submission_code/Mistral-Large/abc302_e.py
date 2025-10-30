import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0

    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1

    queries = []
    for _ in range(Q):
        query = tuple(map(int, data[index:index+3]))
        queries.append(query)
        index += 3

    # Initialize the graph and the degree of each vertex
    graph = {i: set() for i in range(1, N+1)}
    degree = {i: 0 for i in range(1, N+1)}

    # Function to count isolated vertices
    def count_isolated():
        return sum(1 for d in degree.values() if d == 0)

    results = []

    for query in queries:
        if query[0] == 1:
            u, v = query[1], query[2]
            graph[u].add(v)
            graph[v].add(u)
            degree[u] += 1
            degree[v] += 1
        elif query[0] == 2:
            v = query[1]
            for neighbor in graph[v]:
                graph[neighbor].remove(v)
                degree[neighbor] -= 1
            graph[v] = set()
            degree[v] = 0

        results.append(count_isolated())

    for result in results:
        print(result)

if __name__ == "__main__":
    solve()