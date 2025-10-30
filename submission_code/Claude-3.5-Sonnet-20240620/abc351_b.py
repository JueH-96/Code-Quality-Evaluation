# YOUR CODE HERE
def find_different_cell(N, grid_A, grid_B):
    for i in range(N):
        for j in range(N):
            if grid_A[i][j] != grid_B[i][j]:
                return i + 1, j + 1  # Adding 1 to convert 0-based index to 1-based index
    return None  # This should never happen if the input is valid

N = int(input())
grid_A = [input() for _ in range(N)]
grid_B = [input() for _ in range(N)]

result = find_different_cell(N, grid_A, grid_B)
print(f"{result[0]} {result[1]}")