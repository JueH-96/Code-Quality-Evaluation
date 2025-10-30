import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
Q = int(data[index])
index += 1

# Initialize the graph as a list of sets
graph = [set() for _ in range(N)]

# Initialize the count of isolated vertices
isolated_count = N

# Process each query
for _ in range(Q):
    query_type = int(data[index])
    index += 1

    if query_type == 1:
        u = int(data[index]) - 1
        index += 1
        v = int(data[index]) - 1
        index += 1

        if not graph[u]:
            isolated_count -= 1
        if not graph[v]:
            isolated_count -= 1

        graph[u].add(v)
        graph[v].add(u)

    elif query_type == 2:
        v = int(data[index]) - 1
        index += 1

        if graph[v]:
            for neighbor in graph[v]:
                graph[neighbor].remove(v)
                if not graph[neighbor]:
                    isolated_count += 1
            isolated_count += 1
            graph[v].clear()

    print(isolated_count)