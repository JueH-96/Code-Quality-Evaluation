def solve():
    n, m = map(int, input().split())
    relationships = []
    for _ in range(m):
        relationships.append(list(map(int, input().split())))
    
    stronger = [[False for _ in range(n)] for _ in range(n)]
    for rel in relationships:
        u, v = rel
        stronger[u-1][v-1] = True
        
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i != j and i != k and j != k and stronger[i][k] and stronger[k][j]:
                    stronger[i][j] = True
                    
    strongest_candidates = []
    for i in range(n):
        is_strongest = True
        for j in range(n):
            if i == j:
                continue
            if not stronger[i][j]:
                is_strongest = False
                break
        if is_strongest:
            strongest_candidates.append(i+1)
            
    if len(strongest_candidates) == 1:
        print(strongest_candidates[0])
    else:
        print("-1")

if __name__ == '__main__':
    solve()