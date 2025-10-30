n = int(input())
A = list(map(int, input().split()))
visited = [False] * (n + 1)

for i in range(1, n + 1):
    if not visited[i]:
        path = []
        indices = {}
        current_node = i
        while True:
            if current_node in indices:
                # Found a cycle
                start = indices[current_node]
                cycle = path[start:]
                print(len(cycle))
                print(' '.join(map(str, cycle)))
                exit()
            if visited[current_node]:
                break
            visited[current_node] = True
            indices[current_node] = len(path)
            path.append(current_node)
            current_node = A[current_node - 1]