import sys

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1
B = [int(x) for x in data[index:index + N]]

MOD = 998244353
INF = 2001

# Compute prefix sum of free positions
prefix_free = [0] * (N + 1)
for pos in range(1, N + 1):
    if B[pos - 1] == -1:
        prefix_free[pos] = prefix_free[pos - 1] + 1
    else:
        prefix_free[pos] = prefix_free[pos - 1]

Q = prefix_free[N]  # Total number of free positions

# Compute prefix min of fixed values
min_fixed_prefix = [INF] * (N + 1)
current_min = INF
for k in range(1, N + 1):
    if B[k - 1] != -1:  # Fixed position
        current_min = min(current_min, B[k - 1])
    min_fixed_prefix[k] = current_min  # Min up to position k

# Initialize sum of answer
sum_ans = 0

# Iterate over each position i from 1 to N
for i in range(1, N + 1):
    if B[i - 1] == -1:  # Position i is free
        min_f = min_fixed_prefix[i - 1]
        E = prefix_free[i - 1]  # Number of free positions up to i-1
        if min_f == INF:
            if E == 0:  # Special case for i=1 with no previous positions
                num_fav_prefix = M % MOD
            else:
                sum_s = 0
                for s in range(1, M + 1):
                    sum_s += pow(s, E, MOD)
                    sum_s %= MOD
                pow_M_E = pow(M, E, MOD)
                num_fav_prefix = (sum_s - pow_M_E) % MOD
                if num_fav_prefix < 0:
                    num_fav_prefix += MOD
        else:  # min_f is finite
            lower = M - min_f + 1
            sum_s = 0
            for s in range(lower, M + 1):
                sum_s += pow(s, E, MOD)
                sum_s %= MOD
            pow_M_E = pow(M, E, MOD)
            num_fav_prefix = (sum_s - pow_M_E) % MOD
            if num_fav_prefix < 0:
                num_fav_prefix += MOD
    else:  # Position i is fixed
        c = B[i - 1]
        min_f = min_fixed_prefix[i - 1]
        E = prefix_free[i - 1]
        if min_f >= c + 1:
            base = max(0, M - c)
            num_fav_prefix = pow(base, E, MOD)
        else:
            num_fav_prefix = 0

    # Compute the number for this i
    cum_free_i = prefix_free[i]  # Number of free positions up to i inclusive
    exponent_rem = Q - cum_free_i
    pow_M_rem = pow(M, exponent_rem, MOD)
    num_for_i = (num_fav_prefix * pow_M_rem) % MOD

    # Add to the sum
    sum_ans += num_for_i
    sum_ans %= MOD

# Output the result
print(sum_ans)