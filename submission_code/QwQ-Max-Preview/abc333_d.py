import sys
from sys import stdin
sys.setrecursionlimit(1 << 25)

def main():
    N = int(stdin.readline())
    adj = [[] for _ in range(N + 1)]  # 1-based indexing
    for _ in range(N - 1):
        u, v = map(int, stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    
    # Find children of vertex 1
    children = []
    for neighbor in adj[1]:
        if neighbor != 1:  # since it's undirected, 1's neighbors include itself? No, but input ensures u < v, so 1's neighbors are correct
            children.append(neighbor)
    
    max_size = 0
    for c in children:
        # Compute size of subtree rooted at c with parent 1
        stack = [(c, 1)]
        size = 0
        while stack:
            node, parent = stack.pop()
            size += 1
            for neighbor in adj[node]:
                if neighbor != parent:
                    stack.append((neighbor, node))
        if size > max_size:
            max_size = size
    
    print(max_size + 1)

if __name__ == "__main__":
    main()