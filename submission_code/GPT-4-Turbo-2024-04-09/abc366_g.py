def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    from collections import defaultdict
    
    edges = defaultdict(list)
    
    index = 2
    for _ in range(M):
        u = int(data[index]) - 1
        v = int(data[index + 1]) - 1
        edges[u].append(v)
        edges[v].append(u)
        index += 2
    
    # If there are no edges, any assignment is valid
    if M == 0:
        print("Yes")
        print(" ".join(["1"] * N))
        return
    
    # Check if the graph is bipartite
    color = [-1] * N
    
    def dfs(node, c):
        if color[node] != -1:
            return color[node] == c
        color[node] = c
        for neighbor in edges[node]:
            if not dfs(neighbor, 1 - c):
                return False
        return True
    
    is_bipartite = True
    for node in range(N):
        if color[node] == -1:
            if not dfs(node, 0):
                is_bipartite = False
                break
    
    if not is_bipartite:
        print("No")
    else:
        # If the graph is bipartite, we can assign values such that all XOR sums are zero
        print("Yes")
        # Assign 1 to one set and 2 to the other set
        result = [1 if color[i] == 0 else 2 for i in range(N)]
        print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()