def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    idx = 1
    
    # Read the triangular matrix into a 2D list A
    # A[i][j] = A_{i+1, j+1}, valid only if i >= j in 0-based indexing
    A = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i+1):
            A[i][j] = int(data[idx])
            idx += 1
    
    # Start with element 1, combine with elements 1..N in order
    current_element = 1
    for i in range(1, N+1):
        if current_element >= i:
            current_element = A[current_element - 1][i - 1]
        else:
            current_element = A[i - 1][current_element - 1]
    
    print(current_element)

# Do not forget to call main()!
if __name__ == "__main__":
    main()