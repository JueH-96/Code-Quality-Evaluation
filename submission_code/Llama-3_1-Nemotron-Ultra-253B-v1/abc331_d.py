import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    prefix_sum = []
    for _ in range(N):
        row = input[ptr]
        ptr += 1
        ps = [0] * (N + 1)
        for i in range(N):
            ps[i+1] = ps[i] + (1 if row[i] == 'B' else 0)
        prefix_sum.append(ps)

    for _ in range(Q):
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        D = int(input[ptr])
        ptr += 1

        # Compute column parameters
        L_col = D - B + 1
        full_cycles_col = L_col // N
        rem_col = L_col % N
        start_col = B % N
        end_col = start_col + rem_col - 1

        # Compute row_contribution for each row
        sum_total = 0
        row_contribution = []
        for r in range(N):
            cnt = full_cycles_col * prefix_sum[r][N]
            if rem_col > 0:
                if end_col < N:
                    cnt += prefix_sum[r][end_col + 1] - prefix_sum[r][start_col]
                else:
                    cnt += (prefix_sum[r][N] - prefix_sum[r][start_col]) + prefix_sum[r][(end_col % N) + 1]
            row_contribution.append(cnt)
            sum_total += cnt

        # Compute row parameters
        total_rows = C - A + 1
        full_cycles_row = total_rows // N
        rem_row = total_rows % N
        start_row = A % N

        sum_rem = 0
        for i in range(rem_row):
            r = (start_row + i) % N
            sum_rem += row_contribution[r]

        total = full_cycles_row * sum_total + sum_rem
        print(total)

if __name__ == '__main__':
    main()