def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:1+N*2:2]))
    B = list(map(int, data[2:1+N*2:2]))
    
    adj = [[] for _ in range(N)]
    
    for i in range(N):
        for j in range(i+1, N):
            if A[i] == A[j] or B[i] == B[j]:
                adj[i].append(j)
                adj[j].append(i)
    
    visited = [False] * N
    X = 0
    
    def dfs(node):
        visited[node] = True
        component.append(node)
        for neighbor in adj[node]:
            if not visited[neighbor]:
                dfs(neighbor)
    
    for i in range(N):
        if not visited[i]:
            component = []
            dfs(i)
            n = len(component)
            moves = n // 2
            X ^= moves
    
    if X != 0:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()