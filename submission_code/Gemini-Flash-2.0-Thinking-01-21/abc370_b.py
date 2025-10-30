def solve():
    n = int(input())
    matrix_a = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        row_values = list(map(int, input().split()))
        for j in range(1, i + 1):
            matrix_a[i][j] = row_values[j - 1]
    
    current_element = 1
    for e in range(1, n + 1):
        element1 = current_element
        element2 = e
        if element1 >= element2:
            result_element = matrix_a[element1][element2]
        else:
            result_element = matrix_a[element2][element1]
        current_element = result_element
        
    print(current_element)

if __name__ == '__main__':
    solve()