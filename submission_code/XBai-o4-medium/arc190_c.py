import sys

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr])
    ptr += 1
    W = int(input[ptr])
    ptr += 1

    # Read grid A
    A = [[0] * (W + 1) for _ in range(H + 1)]
    for h in range(1, H + 1):
        for w in range(1, W + 1):
            A[h][w] = int(input[ptr]) % MOD
            ptr += 1

    # Compute initial dp
    dp = [[0] * (W + 1) for _ in range(H + 1)]
    dp[1][1] = A[1][1] % MOD
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            if i == 1 and j == 1:
                continue
            up = dp[i-1][j] if i > 1 else 0
            left = dp[i][j-1] if j > 1 else 0
            dp[i][j] = ((up + left) * A[i][j]) % MOD

    Q = int(input[ptr])
    ptr += 1
    sh = int(input[ptr])
    ptr += 1
    sw = int(input[ptr])
    ptr += 1
    cur_h, cur_w = sh, sw

    for _ in range(Q):
        direction = input[ptr]
        ptr += 1
        a_i = int(input[ptr])
        ptr += 1

        # Compute new_h and new_w
        if direction == 'L':
            new_h = cur_h
            new_w = cur_w - 1
        elif direction == 'R':
            new_h = cur_h
            new_w = cur_w + 1
        elif direction == 'U':
            new_h = cur_h - 1
            new_w = cur_w
        else:  # D
            new_h = cur_h + 1
            new_w = cur_w

        # Update A
        A[new_h][new_w] = a_i % MOD
        h, w = new_h, new_w

        start_diagonal = h + w - 1
        end_diagonal = H + W - 1

        for k in range(start_diagonal, end_diagonal + 1):
            i_low = max(h, (k + 1) - W)
            i_high = min(H, (k + 1) - w)
            if i_low > i_high:
                continue
            for i in range(i_low, i_high + 1):
                j = (k + 1) - i
                up = dp[i-1][j] if i > 1 else 0
                left = dp[i][j-1] if j > 1 else 0
                dp[i][j] = ((up + left) * A[i][j]) % MOD

        # Output
        print(dp[H][W] % MOD)
        # Update current position
        cur_h, cur_w = new_h, new_w

if __name__ == '__main__':
    main()