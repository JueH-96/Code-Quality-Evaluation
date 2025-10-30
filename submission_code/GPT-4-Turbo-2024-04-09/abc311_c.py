def find_cycle(N, A):
    visited = [False] * (N + 1)
    in_cycle = [False] * (N + 1)
    cycle = []
    
    def dfs(v):
        nonlocal cycle_found
        if cycle_found:
            return
        stack = []
        path = []
        current = v
        indices = {}
        
        while True:
            if visited[current]:
                if current in indices:
                    # Cycle found
                    start_index = indices[current]
                    cycle.extend(path[start_index:])
                    cycle_found = True
                break
            visited[current] = True
            stack.append(current)
            path.append(current)
            indices[current] = len(path) - 1
            current = A[current - 1]
    
    cycle_found = False
    for i in range(1, N + 1):
        if not visited[i]:
            dfs(i)
        if cycle_found:
            break
    
    print(len(cycle))
    print(" ".join(map(str, cycle)))

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))
find_cycle(N, A)