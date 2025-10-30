import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    grid = []
    for _ in range(N):
        s = input[ptr]
        ptr += 1
        grid.append([1 if c == 'B' else 0 for c in s])

    # Compute prefix sum
    pre = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(1, N+1):
            pre[i][j] = grid[i-1][j-1] + pre[i-1][j] + pre[i][j-1] - pre[i-1][j-1]

    total_black = pre[N][N]

    def f(x, y):
        if x < 0 or y < 0:
            return 0
        full_x = (x + 1) // N
        rem_x = (x + 1) % N
        full_y = (y + 1) // N
        rem_y = (y + 1) % N

        total = full_x * full_y * total_black

        if rem_x > 0:
            sum_rows = pre[rem_x][N]
            total += full_y * sum_rows

        if rem_y > 0:
            sum_cols = pre[N][rem_y]
            total += full_x * sum_cols

        if rem_x > 0 and rem_y > 0:
            sum_partial = pre[rem_x][rem_y]
            total += sum_partial

        return total

    output = []
    for _ in range(Q):
        A = int(input[ptr])
        ptr += 1
        B = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        D = int(input[ptr])
        ptr += 1

        # Compute the area using inclusion-exclusion
        count = f(C, D) - f(A - 1, D) - f(C, B - 1) + f(A - 1, B - 1)
        output.append(str(count))

    print('
'.join(output))

if __name__ == "__main__":
    main()