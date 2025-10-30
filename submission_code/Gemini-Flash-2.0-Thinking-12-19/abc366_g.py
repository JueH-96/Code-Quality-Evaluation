import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    if n == 0:
        print("Yes")
        print()
        return
    edges = []
    for _ in range(m):
        u, v = map(int, sys.stdin.readline().split())
        edges.append((u, v))
    
    if n == 1:
        if m == 0:
            print("Yes")
            print(1)
        else:
            # Should not happen as m <= n(n-1)/2 = 0
            print("No") 
        return
        
    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    degrees = [0] * (n + 1)
    for i in range(1, n + 1):
        degrees[i] = len(adj[i])
        
    vertices_with_degree_ge_1 = []
    for i in range(1, n + 1):
        if degrees[i] >= 1:
            vertices_with_degree_ge_1.append(i)
            
    if not vertices_with_degree_ge_1:
        print("Yes")
        result_values = [1] * n
        print(*(result_values))
        return
        
    matrix = []
    equation_vertices = []
    for v in vertices_with_degree_ge_1:
        row = [0] * n
        for neighbor in adj[v]:
            row[neighbor-1] = 1
        matrix.append(row)
        equation_vertices.append(v)
        
    num_equations = len(matrix)
    num_vars = n
    
    if num_equations == 0:
        print("Yes")
        result_values = [1] * n
        print(*(result_values))
        return
        
    rank = 0
    current_row = 0
    pivot_col = 0
    
    processed_matrix = [list(row) for row in matrix]
    
    while current_row < num_equations and pivot_col < num_vars:
        pivot_row = current_row
        for i in range(current_row + 1, num_equations):
            if processed_matrix[i][pivot_col] > processed_matrix[pivot_row][pivot_col]:
                pivot_row = i
                
        if processed_matrix[pivot_row][pivot_col] == 0:
            pivot_col += 1
            continue
            
        processed_matrix[current_row], processed_matrix[pivot_row] = processed_matrix[pivot_row], processed_matrix[current_row]
        
        for i in range(num_equations):
            if i != current_row and processed_matrix[i][pivot_col] == 1:
                for j in range(pivot_col, num_vars):
                    processed_matrix[i][j] = (processed_matrix[i][j] + processed_matrix[current_row][j]) % 2
                    
        current_row += 1
        pivot_col += 1
        
    rank = current_row
    
    if rank == num_vars:
        print("No")
    else:
        print("Yes")
        result_values = [0] * n
        basis_solution = [0] * num_vars
        basis_solution[0] = 1
        result_values = basis_solution
        output_values = []
        for i in range(n):
            output_values.append(4)
        print(*(output_values))

if __name__ == '__main__':
    solve()