import sys

def find_cycle(n, edges):
    visited = [False] * (n + 1)
    path = [False] * (n + 1)
    for i in range(1, n + 1):
        if not visited[i]:
            if dfs(i, visited, path, edges):
                return i
    return -1

def dfs(node, visited, path, edges):
    visited[node] = True
    path[node] = True
    next_node = edges[node]
    if path[next_node]:
        return True
    if not visited[next_node] and dfs(next_node, visited, path, edges):
        return True
    path[node] = False
    return False

def main():
    n = int(sys.stdin.readline().strip())
    edges = [0] + list(map(int, sys.stdin.readline().strip().split()))
    start = find_cycle(n, edges)
    if start == -1:
        print("No solution")
        return
    cycle = [start]
    current = edges[start]
    while current != start:
        cycle.append(current)
        current = edges[current]
    print(len(cycle))
    print(' '.join(map(str, cycle)))

if __name__ == "__main__":
    main()