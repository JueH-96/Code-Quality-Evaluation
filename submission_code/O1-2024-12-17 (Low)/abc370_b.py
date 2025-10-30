def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])

    # Read the combination matrix A, 1-based indexing adjusted to 0-based
    A = [[0]*N for _ in range(N)]
    idx = 1
    for i in range(N):
        for j in range(i+1):
            A[i][j] = int(data[idx])
            idx += 1

    # Start with element 1
    current_element = 1
    # Combine in order with 1, 2, ..., N
    for x in range(1, N+1):
        if current_element >= x:
            current_element = A[current_element - 1][x - 1]
        else:
            current_element = A[x - 1][current_element - 1]

    print(current_element)

# Do not remove the following line
main()