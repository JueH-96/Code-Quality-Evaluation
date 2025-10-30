# YOUR CODE HERE
import sys

def preprocess(N, P):
    # Precompute the prefix sum of black squares
    prefix_sum = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(N):
        for j in range(N):
            prefix_sum[i + 1][j + 1] = prefix_sum[i + 1][j] + prefix_sum[i][j + 1] - prefix_sum[i][j] + (P[i][j] == 'B')
    return prefix_sum

def count_black_squares(prefix_sum, A, B, C, D, N):
    # Calculate the number of black squares in the given rectangle
    rows = (C - A + 1) // N
    cols = (D - B + 1) // N
    extra_rows = (C - A + 1) % N
    extra_cols = (D - B + 1) % N

    full_blocks = rows * cols * prefix_sum[N][N]
    extra_cols_blocks = rows * prefix_sum[N][extra_cols]
    extra_rows_blocks = cols * prefix_sum[extra_rows][N]
    extra_block = prefix_sum[extra_rows][extra_cols]

    return full_blocks + extra_cols_blocks + extra_rows_blocks + extra_block

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    P = [data[2 + i * N: 2 + (i + 1) * N] for i in range(N)]
    
    prefix_sum = preprocess(N, P)
    
    results = []
    for i in range(Q):
        A, B, C, D = map(int, data[2 + N * N + i * 4: 2 + N * N + (i + 1) * 4])
        result = count_black_squares(prefix_sum, A, B, C, D, N)
        results.append(result)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()