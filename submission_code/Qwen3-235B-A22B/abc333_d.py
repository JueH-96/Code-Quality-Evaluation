import sys
sys.setrecursionlimit(1 << 25)
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    data = list(map(int, input[1:]))
    adj = defaultdict(list)
    for i in range(0, len(data), 2):
        u = data[i]
        v = data[i+1]
        adj[u].append(v)
        adj[v].append(u)
    
    # Handle special case if node 1 has degree 1
    children_of_1 = adj[1]
    if len(children_of_1) <= 1:
        print(1)
        return
    
    # Compute depth for each subtree
    def compute_depth(root):
        stack = [(root, -1, False)]
        max_depth = 0
        depth = {}
        while stack:
            node, parent, visited = stack.pop()
            if visited:
                current_max = 0
                for neighbor in adj[node]:
                    if neighbor != parent:
                        current_max = max(current_max, depth[neighbor])
                depth[node] = current_max + 1
            else:
                stack.append((node, parent, True))
                for neighbor in adj[node]:
                    if neighbor != parent:
                        stack.append((neighbor, node, False))
        return depth[root]
    
    total = 0
    for child in adj[1]:
        # Temporarily remove the connection to node 1 to avoid cycles
        # and compute depth starting from child
        total += compute_depth(child)
    
    print(total)

if __name__ == "__main__":
    main()