import sys
sys.setrecursionlimit(10000)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    K = int(data[2])
    S = data[3:3+H]
    
    grid = [list(S[i]) for i in range(H)]
    
    empty_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                empty_cells.append((i, j))
    
    num_empty = len(empty_cells)
    if num_empty < K + 1:
        print(0)
        return
    
    # Precompute adjacency list
    adj = []
    for i in range(H):
        adj.append([])
        for j in range(W):
            if grid[i][j] == '.':
                neighbors = []
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.':
                        neighbors.append((ni, nj))
                adj[i].append(neighbors)
            else:
                adj[i].append([])
    
    count = 0
    
    def dfs(i, j, moves_left, visited):
        nonlocal count
        if moves_left == 0:
            count += 1
            return
        for ni, nj in adj[i][j]:
            if (ni, nj) not in visited:
                visited.add((ni, nj))
                dfs(ni, nj, moves_left - 1, visited)
                visited.remove((ni, nj))
    
    for start_i, start_j in empty_cells:
        visited = set()
        visited.add((start_i, start_j))
        dfs(start_i, start_j, K, visited)
    
    print(count)

if __name__ == '__main__':
    main()