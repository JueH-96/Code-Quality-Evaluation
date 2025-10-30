from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = [[] for _ in range(N+1)]
    index = 1
    for _ in range(N-1):
        u = int(data[index])
        v = int(data[index+1])
        edges[u].append(v)
        edges[v].append(u)
        index += 2
    
    # BFS to find the depth of each node from node 1
    depth = [0] * (N+1)
    visited = [False] * (N+1)
    q = deque()
    q.append(1)
    visited[1] = True
    
    while q:
        node = q.popleft()
        for neighbor in edges[node]:
            if not visited[neighbor]:
                depth[neighbor] = depth[node] + 1
                visited[neighbor] = True
                q.append(neighbor)
    
    # Now, we need to find the maximum depth in the subtree rooted at 1
    # Since the tree is rooted at 1, the depth is already computed
    # The number of operations is the maximum depth + 1
    # Because we need to delete all nodes on the path from the deepest leaf to 1
    max_depth = max(depth)
    print(max_depth + 1)

if __name__ == "__main__":
    main()