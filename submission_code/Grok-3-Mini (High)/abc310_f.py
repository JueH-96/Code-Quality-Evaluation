import sys

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))
index += N

mod = 998244353
MASK_SIZE = 2048  # 2^11

# Initialize DP for 0 dice: only sum 0 is achievable
dp_curr = [0] * MASK_SIZE
dp_curr[1] = 1  # Mask with only sum 0 set

# Add each die one by one
for i in range(N):
    A_next = A[i]
    dp_next = [0] * MASK_SIZE
    for old_mask in range(MASK_SIZE):
        dp_val = dp_curr[old_mask]
        if dp_val == 0:
            continue  # Skip if no ways to achieve this mask
        # Handle small values v <= 10
        num_small = min(A_next, 10)
        for k in range(1, num_small + 1):
            new_mask = (old_mask | (old_mask << k)) & 2047
            dp_next[new_mask] = (dp_next[new_mask] + dp_val) % mod
        # Handle large values v > 10
        if A_next > 10:
            num_large_v = A_next - 10
            add_amount = (dp_val * num_large_v) % mod
            dp_next[old_mask] = (dp_next[old_mask] + add_amount) % mod
    # Update dp_curr for next iteration
    dp_curr = dp_next

# Compute sum of favorable outcomes (where sum 10 is achievable)
sum_fav = 0
for mask in range(MASK_SIZE):
    if mask & (1 << 10):  # Check if bit 10 is set
        sum_fav += dp_curr[mask]
        sum_fav %= mod

# Compute total number of outcomes modulo mod
total_prod = 1
for a in A:
    total_prod = (total_prod * a) % mod

# Compute modular inverse of total_prod
inv_total = pow(total_prod, mod - 2, mod)

# Compute probability modulo mod
prob = (sum_fav * inv_total) % mod

# Output the result
print(prob)