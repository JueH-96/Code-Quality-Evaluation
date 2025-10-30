n = int(input())
A = list(map(int, input().split()))
visited = [False] * (n + 1)

for i in range(1, n + 1):
    if not visited[i]:
        current_path = []
        current_path_dict = {}
        current_node = i
        while True:
            if visited[current_node]:
                if current_node in current_path_dict:
                    start_index = current_path_dict[current_node]
                    cycle = current_path[start_index:]
                    print(len(cycle))
                    print(' '.join(map(str, cycle)))
                    exit()
                else:
                    break
            else:
                visited[current_node] = True
                current_path.append(current_node)
                current_path_dict[current_node] = len(current_path) - 1
                current_node = A[current_node - 1]