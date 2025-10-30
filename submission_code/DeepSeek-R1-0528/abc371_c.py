import itertools

def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    N = int(data[0].strip())
    index = 1
    
    M_G = int(data[index].strip())
    index += 1
    adjG = [[0] * N for _ in range(N)]
    for i in range(M_G):
        u, v = map(int, data[index].split())
        index += 1
        u -= 1
        v -= 1
        adjG[u][v] = 1
        adjG[v][u] = 1
        
    M_H = int(data[index].strip())
    index += 1
    adjH = [[0] * N for _ in range(N)]
    for i in range(M_H):
        a, b = map(int, data[index].split())
        index += 1
        a -= 1
        b -= 1
        adjH[a][b] = 1
        adjH[b][a] = 1
        
    cost = [[0] * N for _ in range(N)]
    for i in range(0, N-1):
        row = data[index].split()
        index += 1
        for k in range(len(row)):
            j = i + 1 + k
            c = int(row[k])
            cost[i][j] = c
            cost[j][i] = c

    perms = itertools.permutations(range(N))
    ans = float('inf')
    
    for p in perms:
        q = [0] * N
        for idx in range(N):
            q[p[idx]] = idx
        
        total_cost = 0
        for i in range(N):
            for j in range(i+1, N):
                if adjH[i][j] != adjG[q[i]][q[j]]:
                    total_cost += cost[i][j]
                    
        if total_cost < ans:
            ans = total_cost
            
    print(ans)

if __name__ == "__main__":
    main()