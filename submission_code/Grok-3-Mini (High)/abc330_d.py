import sys

# Read input from stdin
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
S = []
for _ in range(N):
    S.append(data[index])
    index += 1

# Compute the sum of 'o's for each column
col_sum = [0] * N
for i in range(N):
    for j in range(N):
        if S[i][j] == 'o':
            col_sum[j] += 1

# Initialize answer
answer = 0

# For each row, compute the contribution to the answer
for i in range(N):
    K = 0  # Number of 'o's in the row
    sum_col_sum = 0  # Sum of col_sum[j] for columns with 'o' in this row
    for j in range(N):
        if S[i][j] == 'o':
            K += 1
            sum_col_sum += col_sum[j]
    if K > 1:
        answer += (K - 1) * (sum_col_sum - K)

# Output the answer
print(answer)