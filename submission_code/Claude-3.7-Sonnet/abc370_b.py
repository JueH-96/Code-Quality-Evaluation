def solve():
    N = int(input())
    
    # Reading the transformation matrix
    A = []
    for i in range(N):
        row = list(map(int, input().split()))
        A.append(row)
    
    # Perform the combinations
    current_element = 1  # Start with element 1
    for j in range(1, N+1):
        if current_element >= j:
            current_element = A[current_element-1][j-1]
        else:
            current_element = A[j-1][current_element-1]
    
    return current_element

print(solve())