import sys

def main():
    """
    Main function to solve the problem.
    """
    sys.setrecursionlimit(2 * 10**5 + 5)
    MOD = 998244353

    try:
        input = sys.stdin.readline
    except:
        input = lambda: ""

    H, W = map(int, input().split())
    A = [list(map(int, input().split())) for _ in range(H)]
    Q_val, sh, sw = map(int, input().split())
    sh -= 1
    sw -= 1

    queries = [line.split() for line in sys.stdin.read().splitlines()]

    dp_start = [[0] * W for _ in range(H)]
    dp_end = [[0] * W for _ in range(H)]

    def compute_dp_start_from_scratch():
        for r in range(H):
            for c in range(W):
                val = 0
                if r == 0 and c == 0:
                    val = 1
                if r > 0:
                    val = (val + dp_start[r - 1][c]) % MOD
                if c > 0:
                    val = (val + dp_start[r][c - 1]) % MOD
                dp_start[r][c] = (val * A[r][c]) % MOD
    
    def compute_dp_end_from_scratch():
        for r in range(H - 1, -1, -1):
            for c in range(W - 1, -1, -1):
                val = 0
                if r == H - 1 and c == W - 1:
                    val = 1
                if r < H - 1:
                    val = (val + dp_end[r + 1][c]) % MOD
                if c < W - 1:
                    val = (val + dp_end[r][c + 1]) % MOD
                dp_end[r][c] = (val * A[r][c]) % MOD

    compute_dp_start_from_scratch()
    compute_dp_end_from_scratch()
    
    total_sum = dp_start[H - 1][W - 1]

    moves = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
    
    current_h, current_w = sh, sw
    
    for d_char, a_val_str in queries:
        a_val = int(a_val_str)
        
        dh, dw = moves[d_char]
        current_h += dh
        current_w += dw
        
        r, c = current_h, current_w
        
        old_val = A[r][c]
        
        if old_val == a_val:
            sys.stdout.write(str(total_sum) + '
')
            continue
            
        sum_to_prec = 0
        if r == 0 and c == 0:
            sum_to_prec = 1
        else:
            if r > 0:
                sum_to_prec = (sum_to_prec + dp_start[r - 1][c]) % MOD
            if c > 0:
                sum_to_prec = (sum_to_prec + dp_start[r][c - 1]) % MOD

        sum_from_succ = 0
        if r == H - 1 and c == W - 1:
            sum_from_succ = 1
        else:
            if r < H - 1:
                sum_from_succ = (sum_from_succ + dp_end[r + 1][c]) % MOD
            if c < W - 1:
                sum_from_succ = (sum_from_succ + dp_end[r][c + 1]) % MOD
            
        k1 = (sum_to_prec * sum_from_succ) % MOD
        
        diff = (a_val - old_val + MOD) % MOD
        total_sum = (total_sum + diff * k1) % MOD
        
        A[r][c] = a_val
        sys.stdout.write(str(total_sum) + '
')
        
        # Update DP tables
        # Update dp_start from row r downwards
        for i in range(r, H):
            start_j = c if i == r else 0
            for j in range(start_j, W):
                val = 0
                if i == 0 and j == 0:
                    val = 1
                if i > 0:
                    val = (val + dp_start[i - 1][j]) % MOD
                if j > 0:
                    val = (val + dp_start[i][j - 1]) % MOD
                dp_start[i][j] = (val * A[i][j]) % MOD

        # Update dp_end from row r upwards
        for i in range(r, -1, -1):
            end_j = c if i == r else W - 1
            for j in range(end_j, -1, -1):
                val = 0
                if i == H - 1 and j == W - 1:
                    val = 1
                if i < H - 1:
                    val = (val + dp_end[i + 1][j]) % MOD
                if j < W - 1:
                    val = (val + dp_end[i][j + 1]) % MOD
                dp_end[i][j] = (val * A[i][j]) % MOD

if __name__ == "__main__":
    main()