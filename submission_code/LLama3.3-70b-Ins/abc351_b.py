def find_difference():
    # Read the size of the grid
    N = int(input())

    # Read grid A
    grid_A = []
    for _ in range(N):
        row = input()
        grid_A.append(row)

    # Read grid B
    grid_B = []
    for _ in range(N):
        row = input()
        grid_B.append(row)

    # Find the difference
    for i in range(N):
        for j in range(N):
            if grid_A[i][j] != grid_B[i][j]:
                return i + 1, j + 1

    # If no difference is found, return None
    return None

# Get the result
result = find_difference()

# Print the result
if result:
    print(result[0], result[1])
else:
    print("No difference found")