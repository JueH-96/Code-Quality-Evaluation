import sys
from collections import deque

MOD = 998244353

def main():
    data = sys.stdin.read().splitlines()
    t = int(data[0])
    index = 1
    results = []
    
    for _ in range(t):
        header = data[index].split()
        index += 1
        H = int(header[0])
        W = int(header[1])
        grid = []
        for i in range(H):
            grid.append(data[index].strip())
            index += 1
            
        valid_input = True
        for i in range(H):
            cnt = 0
            for c in grid[i]:
                if c == 'A':
                    cnt += 1
            if cnt % 2 != 0:
                valid_input = False
                break
                
        if not valid_input:
            results.append("0")
            continue
            
        for j in range(W):
            cnt = 0
            for i in range(H):
                if grid[i][j] == 'A':
                    cnt += 1
            if cnt % 2 != 0:
                valid_input = False
                break
                
        if not valid_input:
            results.append("0")
            continue
            
        F_row = []
        for i in range(H):
            arr = [0] * W
            for j in range(1, W):
                add = 1 if grid[i][j-1] == 'A' else 0
                arr[j] = arr[j-1] ^ add
            F_row.append(arr)
            
        F_col = []
        for j in range(W):
            arr = [0] * H
            for i in range(1, H):
                add = 1 if grid[i-1][j] == 'A' else 0
                arr[i] = arr[i-1] ^ add
            F_col.append(arr)
            
        n_nodes = H + W
        graph = [[] for _ in range(n_nodes)]
        
        for i in range(H):
            for j in range(W):
                if grid[i][j] == 'B':
                    rp = F_row[i][j-1] if j >= 1 else 0
                    cp = F_col[j][i-1] if i >= 1 else 0
                    X = rp ^ cp
                    weight = 1 ^ X
                    u = i
                    v = H + j
                    graph[u].append((v, weight))
                    graph[v].append((u, weight))
                    
        vis = [False] * n_nodes
        comp_count = 0
        valid_graph = True
        values = [-1] * n_nodes
        
        for i in range(n_nodes):
            if not vis[i] and valid_graph:
                comp_count += 1
                vis[i] = True
                values[i] = 0
                queue = deque([i])
                while queue and valid_graph:
                    u = queue.popleft()
                    for (v, w) in graph[u]:
                        if not vis[v]:
                            vis[v] = True
                            values[v] = values[u] ^ w
                            queue.append(v)
                        else:
                            if values[v] != (values[u] ^ w):
                                valid_graph = False
                                break
                if not valid_graph:
                    break
                    
        if not valid_graph:
            results.append("0")
        else:
            ans = pow(2, comp_count, MOD)
            results.append(str(ans))
            
    for res in results:
        print(res)

if __name__ == "__main__":
    main()