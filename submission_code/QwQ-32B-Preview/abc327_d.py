from collections import deque

def is_bipartite(graph, colors, start):
    colors[start] = 0
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if colors[neighbor] == -1:
                colors[neighbor] = 1 - colors[node]
                queue.append(neighbor)
            elif colors[neighbor] == colors[node]:
                return False
    return True

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+M]))
    B = list(map(int, data[2+M:2+2*M]))
    
    # Check if any A_i == B_i
    for i in range(M):
        if A[i] == B[i]:
            print("No")
            return
    
    # Build graph
    graph = [[] for _ in range(N+1)]
    for i in range(M):
        graph[A[i]].append(B[i])
        graph[B[i]].append(A[i])
    
    # BFS to check bipartition
    colors = [-1] * (N+1)
    for node in range(1, N+1):
        if colors[node] == -1:
            if not is_bipartite(graph, colors, node):
                print("No")
                return
    print("Yes")

if __name__ == "__main__":
    main()