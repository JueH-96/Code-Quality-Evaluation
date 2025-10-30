# YOUR CODE HERE
def solve():
    N = int(input())

    grid_A = []
    for _ in range(N):
        grid_A.append(input())

    grid_B = []
    for _ in range(N):
        grid_B.append(input())

    found_diff = False
    for i in range(N):
        for j in range(N):
            if grid_A[i][j] != grid_B[i][j]:
                # Found the differing cell. Print 1-based indices and exit.
                print(f"{i + 1} {j + 1}")
                found_diff = True
                break # Exit inner loop
        if found_diff:
            break # Exit outer loop

if __name__ == '__main__':
    solve()