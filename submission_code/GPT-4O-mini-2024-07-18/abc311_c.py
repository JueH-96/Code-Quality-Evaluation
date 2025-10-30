def find_cycle(N, A):
    visited = [False] * (N + 1)
    stack = []
    
    for start in range(1, N + 1):
        if not visited[start]:
            current = start
            path = []
            while not visited[current]:
                visited[current] = True
                path.append(current)
                current = A[current - 1]
            
            if current in path:
                cycle_start_index = path.index(current)
                cycle = path[cycle_start_index:]
                return cycle

N = int(input().strip())
A = list(map(int, input().strip().split()))

cycle = find_cycle(N, A)
print(len(cycle))
print(' '.join(map(str, cycle)))