def count_black_squares(N, Q, P, queries):
    # Precompute the number of black squares in a prefix sum array
    prefix_sum = [[0] * (N + 1) for _ in range(N + 1)]
    
    for i in range(N):
        for j in range(N):
            prefix_sum[i + 1][j + 1] = prefix_sum[i][j + 1] + prefix_sum[i + 1][j] - prefix_sum[i][j] + (1 if P[i][j] == 'B' else 0)

    results = []
    
    for A, B, C, D in queries:
        # Calculate the dimensions of the rectangle
        rect_width = (C - A + 1)
        rect_height = (D - B + 1)

        # Calculate the number of complete N x N blocks in the rectangle
        full_blocks_x = rect_width // N
        full_blocks_y = rect_height // N
        
        # Calculate the number of black squares in full blocks
        total_black = full_blocks_x * full_blocks_y * prefix_sum[N][N]
        
        # Calculate the remaining parts
        rem_x_start = A % N
        rem_x_end = C % N
        rem_y_start = B % N
        rem_y_end = D % N
        
        # Add the black squares from the remaining rows
        for i in range(rem_x_start, rem_x_end + 1):
            for j in range(rem_y_start, rem_y_end + 1):
                total_black += (1 if P[i][j] == 'B' else 0)

        # Add the black squares from the full rows in the remaining height
        for i in range(rem_x_start):
            for j in range(rem_y_start, rem_y_end + 1):
                total_black += (1 if P[i][j] == 'B' else 0)

        # Add the black squares from the full columns in the remaining width
        for i in range(rem_x_start, rem_x_end + 1):
            for j in range(rem_y_start):
                total_black += (1 if P[i][j] == 'B' else 0)

        results.append(total_black)

    return results


import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    first_line = data[0].split()
    N = int(first_line[0])
    Q = int(first_line[1])
    
    P = [data[i + 1] for i in range(N)]
    
    queries = []
    for i in range(Q):
        queries.append(tuple(map(int, data[N + 1 + i].split())))
    
    results = count_black_squares(N, Q, P, queries)
    
    print('
'.join(map(str, results)))

if __name__ == "__main__":
    main()