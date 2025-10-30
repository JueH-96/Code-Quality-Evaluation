import sys

MOD = 998244353

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0

    H = int(data[idx])
    W = int(data[idx+1])
    idx +=2

    A = [[0]*(W+2) for _ in range(H+2)]
    for i in range(1, H+1):
        for j in range(1, W+1):
            A[i][j] = int(data[idx])
            idx +=1

    # Compute prefix
    prefix = [[0]*(W+2) for _ in range(H+2)]
    prefix[1][1] = 1
    for i in range(1, H+1):
        for j in range(1, W+1):
            if i == 1 and j == 1:
                continue
            res = 0
            if i > 1:
                res += prefix[i-1][j] * A[i-1][j]
                if res >= MOD or res < 0:
                    res %= MOD
            if j > 1:
                res += prefix[i][j-1] * A[i][j-1]
                if res >= MOD or res < 0:
                    res %= MOD
            prefix[i][j] = res % MOD

    # Compute suffix
    suffix = [[0]*(W+2) for _ in range(H+2)]
    suffix[H][W] = 1
    for i in range(H, 0, -1):
        for j in range(W, 0, -1):
            if i == H and j == W:
                continue
            res = 0
            if i < H:
                res += suffix[i+1][j] * A[i+1][j]
                if res >= MOD or res < 0:
                    res %= MOD
            if j < W:
                res += suffix[i][j+1] * A[i][j+1]
                if res >= MOD or res < 0:
                    res %= MOD
            suffix[i][j] = res % MOD

    # Compute initial S
    S = (prefix[H][W] * A[H][W]) % MOD

    Q = int(data[idx])
    sh = int(data[idx+1])
    sw = int(data[idx+2])
    idx +=3

    current_h = sh
    current_w = sw

    output = []
    for _ in range(Q):
        d = data[idx]
        a_i = int(data[idx+1])
        idx +=2

        # Compute new_h and new_w
        new_h, new_w = current_h, current_w
        if d == 'U':
            new_h -= 1
        elif d == 'D':
            new_h += 1
        elif d == 'L':
            new_w -= 1
        elif d == 'R':
            new_w += 1

        # Get old value
        old_val = A[new_h][new_w]

        # Compute delta
        delta_part = (a_i - old_val) % MOD
        pr = prefix[new_h][new_w]
        su = suffix[new_h][new_w]
        delta = (pr * su) % MOD
        delta = (delta * delta_part) % MOD
        S = (S + delta) % MOD
        if S < 0:
            S += MOD

        # Update A[new_h][new_w]
        A[new_h][new_w] = a_i

        # Output
        output.append(str(S % MOD))

        # Update current position
        current_h, current_w = new_h, new_w

    print('
'.join(output))

if __name__ == '__main__':
    main()