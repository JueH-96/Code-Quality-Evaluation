# YOUR CODE HERE
import sys
from itertools import accumulate

def read_ints(): return map(int, input().split())
def read_int(): return int(input())
def read_str(): return input().strip()

N, Q = read_ints()
P = [read_str() for _ in range(N)]

# Precompute the prefix sum for each N*N block
prefix_sum = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(N):
    for j in range(N):
        prefix_sum[i + 1][j + 1] = prefix_sum[i + 1][j] + prefix_sum[i][j + 1] - prefix_sum[i][j] + (P[i][j] == 'B')

# Function to get the number of black squares in a block
def get_block_count(x1, y1, x2, y2):
    return prefix_sum[x2][y2] - prefix_sum[x2][y1] - prefix_sum[x1][y2] + prefix_sum[x1][y1]

for _ in range(Q):
    A, B, C, D = read_ints()
    count = 0
    # Calculate the number of full blocks in the x and y directions
    full_x_blocks = (C - A) // N
    full_y_blocks = (D - B) // N
    # Calculate the number of black squares in the full blocks
    count += full_x_blocks * full_y_blocks * get_block_count(N, N, N, N)
    # Calculate the number of black squares in the remaining partial blocks
    for i in range(A % N, C % N + 1):
        for j in range(B % N, D % N + 1):
            if i >= A % N and j >= B % N and i <= C % N and j <= D % N:
                count += get_block_count(i, j, i + 1, j + 1)
    print(count)