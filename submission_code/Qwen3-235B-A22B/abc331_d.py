import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    # Read the NxN grid
    cell = [[0] * N for _ in range(N)]
    for i in range(N):
        line = input[ptr]
        ptr += 1
        for j in range(N):
            cell[i][j] = 1 if line[j] == 'B' else 0

    # Precompute row_sums and col_sums
    row_sums = [sum(cell[i]) for i in range(N)]
    col_sums = [0] * N
    for j in range(N):
        total = 0
        for i in range(N):
            total += cell[i][j]
        col_sums[j] = total

    # Precompute prefix sums for rows and columns
    pre_row = [0] * (N + 1)
    for i in range(N):
        pre_row[i + 1] = pre_row[i] + row_sums[i]
    pre_col = [0] * (N + 1)
    for j in range(N):
        pre_col[j + 1] = pre_col[j] + col_sums[j]

    # Precompute 2D prefix sum matrix S
    S = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(N):
        for j in range(N):
            S[i + 1][j + 1] = S[i + 1][j] + S[i][j + 1] - S[i][j] + cell[i][j]

    total_black = pre_row[N]  # Total number of black cells in the NxN grid

    # Process each query
    output = []
    for _ in range(Q):
        A = int(input[ptr])
        B = int(input[ptr + 1])
        C = int(input[ptr + 2])
        D = int(input[ptr + 3])
        ptr += 4

        # Process rows
        row_count = C - A + 1
        row_full = row_count // N
        row_rem_count = row_count % N
        start_r = A % N
        if row_rem_count > 0:
            end_r = (start_r + row_rem_count - 1) % N
        else:
            end_r = -1  # Not used

        sum_row_rem = 0
        if row_rem_count > 0:
            if start_r <= end_r:
                sum_row_rem = pre_row[end_r + 1] - pre_row[start_r]
            else:
                part1 = pre_row[N] - pre_row[start_r]
                part2 = pre_row[end_r + 1] - pre_row[0]
                sum_row_rem = part1 + part2

        # Process columns
        col_count = D - B + 1
        col_full = col_count // N
        col_rem_count = col_count % N
        start_c = B % N
        if col_rem_count > 0:
            end_c = (start_c + col_rem_count - 1) % N
        else:
            end_c = -1  # Not used

        sum_col_rem = 0
        if col_rem_count > 0:
            if start_c <= end_c:
                sum_col_rem = pre_col[end_c + 1] - pre_col[start_c]
            else:
                part1 = pre_col[N] - pre_col[start_c]
                part2 = pre_col[end_c + 1] - pre_col[0]
                sum_col_rem = part1 + part2

        # Compute terms
        term1 = row_full * col_full * total_black
        term2 = row_full * sum_col_rem
        term3 = col_full * sum_row_rem

        # Compute term4
        term4 = 0
        if row_rem_count > 0 and col_rem_count > 0:
            # Compute x_regions
            x_regions = []
            if start_r <= end_r:
                x_regions.append((start_r, end_r))
            else:
                x_regions.append((start_r, N - 1))
                x_regions.append((0, end_r))

            # Compute y_regions
            y_regions = []
            if start_c <= end_c:
                y_regions.append((start_c, end_c))
            else:
                y_regions.append((start_c, N - 1))
                y_regions.append((0, end_c))

            # Sum all rectangles
            for (x1, x2) in x_regions:
                for (y1, y2) in y_regions:
                    sum_rect = S[x2 + 1][y2 + 1] - S[x1][y2 + 1] - S[x2 + 1][y1] + S[x1][y1]
                    term4 += sum_rect

        total = term1 + term2 + term3 + term4
        output.append(str(total))

    print('
'.join(output))

if __name__ == "__main__":
    main()