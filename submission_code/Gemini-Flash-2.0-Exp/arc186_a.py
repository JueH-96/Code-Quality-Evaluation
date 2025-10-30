def solve():
    n, q = map(int, input().split())
    queries = []
    for _ in range(q):
        queries.append(int(input()))

    def is_fixed(matrix, row, col):
        initial_value = matrix[row][col]
        for r in range(n):
            for c in range(n):
                if r == row and c == col:
                    continue
                
                temp_matrix = [row[:] for row in matrix]
                temp_matrix[row][col] = 1 - initial_value
                
                row_sums = [sum(row) for row in temp_matrix]
                col_sums = [sum(temp_matrix[i][j] for i in range(n)) for j in range(n)]
                
                initial_row_sums = [sum(row) for row in matrix]
                initial_col_sums = [sum(matrix[i][j] for i in range(n)) for j in range(n)]
                
                if row_sums == initial_row_sums and col_sums == initial_col_sums:
                    return False
        return True

    def count_fixed(matrix):
        count = 0
        for i in range(n):
            for j in range(n):
                if is_fixed(matrix, i, j):
                    count += 1
        return count

    def check_if_similar(matrix1, matrix2):
        row_sums1 = [sum(row) for row in matrix1]
        col_sums1 = [sum(matrix1[i][j] for i in range(n)) for j in range(n)]
        row_sums2 = [sum(row) for row in matrix2]
        col_sums2 = [sum(matrix2[i][j] for i in range(n)) for j in range(n)]
        
        return row_sums1 == row_sums2 and col_sums1 == col_sums2

    def find_all_similar_matrices(matrix):
        similar_matrices = [matrix]
        
        def generate_matrices(current_matrix, row, col):
            if row == n:
                return
            if col == n:
                generate_matrices(current_matrix, row + 1, 0)
                return
            
            new_matrix = [r[:] for r in current_matrix]
            new_matrix[row][col] = 1 - new_matrix[row][col]
            
            if check_if_similar(matrix, new_matrix) and new_matrix not in similar_matrices:
                similar_matrices.append(new_matrix)
            
            generate_matrices(current_matrix, row, col + 1)
            generate_matrices(new_matrix, row, col + 1)
        
        generate_matrices(matrix, 0, 0)
        return similar_matrices
    
    def count_fixed_elements(matrix):
        fixed_count = 0
        similar_matrices = find_all_similar_matrices(matrix)
        
        for i in range(n):
            for j in range(n):
                is_fixed = True
                for similar_matrix in similar_matrices:
                    if matrix[i][j] != similar_matrix[i][j]:
                        is_fixed = False
                        break
                if is_fixed:
                    fixed_count += 1
        return fixed_count

    for k in queries:
        found = False
        for i in range(2**(n*n)):
            matrix = []
            temp = i
            for r in range(n):
                row = []
                for c in range(n):
                    row.append(temp % 2)
                    temp //= 2
                matrix.append(row)
            
            if count_fixed_elements(matrix) == k:
                found = True
                break
        
        if found:
            print("Yes")
        else:
            print("No")

solve()