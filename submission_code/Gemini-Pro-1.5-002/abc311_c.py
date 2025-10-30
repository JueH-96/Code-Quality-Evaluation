# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    a = [x - 1 for x in a]

    visited = [False] * n
    path = []

    for start_node in range(n):
        if visited[start_node]:
            continue

        current_path = []
        current_node = start_node
        
        while not visited[current_node]:
            visited[current_node] = True
            current_path.append(current_node)
            current_node = a[current_node]
        
        cycle_start_index = -1
        for i in range(len(current_path)):
            if current_path[i] == current_node:
                cycle_start_index = i
                break
        
        if cycle_start_index != -1:
            cycle = current_path[cycle_start_index:]
            print(len(cycle))
            print(*[x + 1 for x in cycle])
            return

solve()