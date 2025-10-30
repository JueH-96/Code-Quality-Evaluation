import sys
data = sys.stdin.read().split()
N = int(data[0])
A_rows = data[1:N+1]
B_rows = data[N+1:2*N+1]

diff_i, diff_j = -1, -1
for i in range(N):
    for j in range(N):
        if A_rows[i][j] != B_rows[i][j]:
            diff_i, diff_j = i, j
            break  # Break the inner loop
    if diff_i != -1:
        break  # Break the outer loop

# Print the 1-based indices
print(diff_i + 1, diff_j + 1)