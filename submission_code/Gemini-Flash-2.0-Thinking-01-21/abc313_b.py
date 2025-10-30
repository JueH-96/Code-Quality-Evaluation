def solve():
    n, m = map(int, input().split())
    given_relations = []
    for _ in range(m):
        u, v = map(int, input().split())
        given_relations.append((u, v))
    
    strongest_candidates = []
    for p in range(1, n + 1):
        relation_matrix = [[0] * n for _ in range(n)]
        for u, v in given_relations:
            relation_matrix[u-1][v-1] = 1
        for q in range(1, n + 1):
            if q != p:
                relation_matrix[p-1][q-1] = 1
                
        transitive_closure_matrix = [row[:] for row in relation_matrix]
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if transitive_closure_matrix[i][k] == 1 and transitive_closure_matrix[k][j] == 1:
                        transitive_closure_matrix[i][j] = 1
                        
        is_valid_strongest = True
        for i in range(n):
            for j in range(i + 1, n):
                if transitive_closure_matrix[i][j] + transitive_closure_matrix[j][i] != 1:
                    is_valid_strongest = False
                    break
            if not is_valid_strongest:
                break
                
        if is_valid_strongest:
            strongest_candidates.append(p)
            
    if len(strongest_candidates) == 1:
        print(strongest_candidates[0])
    else:
        print("-1")

if __name__ == '__main__':
    solve()