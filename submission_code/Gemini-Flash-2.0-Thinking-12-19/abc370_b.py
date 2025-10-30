def solve():
    n = int(input())
    a_matrix = []
    for i in range(n):
        row_values = list(map(int, input().split()))
        a_matrix.append(row_values)
    
    composition_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            composition_matrix[i][j] = a_matrix[i][j]
            
    current_element_index = 0
    
    for i in range(n):
        combine_with_element_index = i
        u_prime = current_element_index
        v_prime = combine_with_element_index
        if u_prime >= v_prime:
            current_element_index = composition_matrix[u_prime][v_prime] - 1
        else:
            current_element_index = composition_matrix[v_prime][u_prime] - 1
            
    print(current_element_index + 1)

if __name__ == '__main__':
    solve()