import itertools

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    index = 1
    mg = int(data[index]); index += 1
    adjG = [[0] * (n+1) for _ in range(n+1)]
    for _ in range(mg):
        u = int(data[index]); v = int(data[index+1]); index += 2
        adjG[u][v] = 1
        adjG[v][u] = 1
        
    mh = int(data[index]); index += 1
    adjH = [[0] * (n+1) for _ in range(n+1)]
    for _ in range(mh):
        u = int(data[index]); v = int(data[index+1]); index += 2
        adjH[u][v] = 1
        adjH[v][u] = 1
        
    cost_mat = [[0] * (n+1) for _ in range(n+1)]
    for i in range(1, n):
        num_values = n - i
        line = data[index:index+num_values]
        index += num_values
        for k in range(num_values):
            j = i + 1 + k
            val = int(line[k])
            cost_mat[i][j] = val
            cost_mat[j][i] = val
            
    perms = list(itertools.permutations(range(1, n+1)))
    best = float('inf')
    
    for perm in perms:
        total_cost = 0
        for i in range(1, n+1):
            for j in range(i+1, n+1):
                u = perm[i-1]
                v = perm[j-1]
                desired = adjG[i][j]
                current = adjH[u][v]
                if desired != current:
                    total_cost += cost_mat[u][v]
        if total_cost < best:
            best = total_cost
            
    print(best)

if __name__ == "__main__":
    main()