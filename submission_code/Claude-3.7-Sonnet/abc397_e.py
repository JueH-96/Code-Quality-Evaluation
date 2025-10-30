def main():
    N, K = map(int, input().split())
    
    adj_list = [[] for _ in range(N * K + 1)]
    
    for _ in range(N * K - 1):
        u, v = map(int, input().split())
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    if can_decompose(N, K, adj_list):
        print("Yes")
    else:
        print("No")

def can_decompose(N, K, adj_list):
    visited = [False] * len(adj_list)
    
    def find_path(start_node):
        path = []
        
        def dfs(node, depth):
            visited[node] = True
            path.append(node)
            
            if depth == K:
                return True
            
            for neighbor in adj_list[node]:
                if not visited[neighbor]:
                    if dfs(neighbor, depth + 1):
                        return True
            
            visited[node] = False
            path.pop()
            return False
        
        if dfs(start_node, 1):
            return path
        return None
    
    def decompose(paths_remaining):
        if paths_remaining == 0:
            return True
        
        for i in range(1, len(adj_list)):
            if not visited[i]:
                path = find_path(i)
                
                if path:
                    if decompose(paths_remaining - 1):
                        return True
                    
                    # Backtrack
                    for node in path:
                        visited[node] = False
        
        return False
    
    return decompose(N)

if __name__ == "__main__":
    main()