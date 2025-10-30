import sys

def main():
    import sys
    tokens = sys.stdin.read().split()
    idx = 0
    N = int(tokens[idx])
    idx += 1
    M = int(tokens[idx])
    idx += 1
    graph = [[] for _ in range(N + 1)]
    for _ in range(M):
        A = int(tokens[idx])
        idx += 1
        B = int(tokens[idx])
        idx += 1
        X = int(tokens[idx])
        idx += 1
        Y = int(tokens[idx])
        idx += 1
        graph[A].append((B, X, Y))
    
    positions = [None] * (N + 1)
    positions[1] = (0, 0)
    visited = [False] * (N + 1)
    visited[1] = True
    stack = [(1, (0, 0))]
    
    while stack:
        node, pos = stack.pop()
        for neighbor, dx, dy in graph[node]:
            if not visited[neighbor]:
                positions[neighbor] = (pos[0] + dx, pos[1] + dy)
                visited[neighbor] = True
                stack.append((neighbor, positions[neighbor]))
    
    for i in range(1, N + 1):
        if positions[i] is not None:
            print(positions[i][0], positions[i][1])
        else:
            print("undecidable")

if __name__ == '__main__':
    main()