def find_cycle_length(graph, start, visited):
    current = start
    length = 0
    while True:
        if visited[current] != -1:
            return length - visited[current], length
        visited[current] = length
        current = graph[current]
        length += 1

def main():
    N = int(input().strip())
    a = list(map(int, input().strip().split()))
    
    graph = {i: a[i] - 1 for i in range(N)}
    visited = [-1] * N
    cycle_lengths = {}
    path_lengths = {}
    
    for i in range(N):
        if visited[i] == -1:
            cycle_length, path_length = find_cycle_length(graph, i, visited)
            cycle_lengths[i] = cycle_length
            path_lengths[i] = path_length
    
    count = 0
    for i in range(N):
        count += path_lengths[i] - cycle_lengths[i] + cycle_lengths[graph[i]]
    
    print(count)

if __name__ == "__main__":
    main()