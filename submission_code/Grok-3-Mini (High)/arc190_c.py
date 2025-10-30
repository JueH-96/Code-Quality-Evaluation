import sys

MOD = 998244353

data = sys.stdin.read().split()
index = 0
H = int(data[index])
index += 1
W = int(data[index])
index += 1

A = [0] * (H * W)
 for i in range(H):
     for j in range(W):
         A[i * W + j] = int(data[index]) % MOD
         index += 1

Q = int(data[index])
index += 1
sh = int(data[index])
index += 1
sw = int(data[index])
index += 1

h_t = sh
w_t = sw

dp = [0] * (H * W)

def compute_dp(row_start, col_start):
    for i in range(row_start, H + 1):
        for j in range(col_start, W + 1):
            if i == 1 and j == 1:
                dp_idx = (i - 1) * W + (j - 1)
                dp[dp_idx] = A[dp_idx] % MOD
            else:
                sum_val = 0
                if i > 1:
                    sum_val += dp[(i - 2) * W + (j - 1)]  # dp[i-1][j]
                if j > 1:
                    sum_val += dp[(i - 1) * W + (j - 2)]  # dp[i][j-1]
                sum_val %= MOD
                dp_idx = (i - 1) * W + (j - 1)
                dp[dp_idx] = (A[dp_idx] * sum_val) % MOD

# Initial DP computation
compute_dp(1, 1)

# List to store answers
ans = []

 for _ in range(Q):
     d = data[index]
     index += 1
     a = int(data[index])
     index += 1

     # Move Takahashi
     if d == 'L':
         w_t -= 1
     elif d == 'R':
         w_t += 1
     elif d == 'U':
         h_t -= 1
     elif d == 'D':
         h_t += 1

     # New position (h_t, w_t)
     # Set A to a
     A_idx = (h_t - 1) * W + (w_t - 1)
     A[A_idx] = a % MOD

     # Recompute DP from (h_t, w_t) to (H, W)
     compute_dp(h_t, w_t)

     # Get S = dp[H][W]
     S = dp[(H - 1) * W + (W - 1)]
     ans.append(str(S))

# Output all answers
print('
'.join(ans))