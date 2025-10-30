import sys

# Read input
data = sys.stdin.read().splitlines()
N = int(data[0].strip())
A_str = data[1].strip().replace(" ", "")  # Remove any spaces and get the binary string
A = [int(c) for c in A_str]  # Convert string to list of integers

# Recursive function to compute dp
def compute_dp(low, size):
    if size == 1:
        bit = A[low]
        current_out = bit
        dp0 = 0 if bit == 0 else 1
        dp1 = 0 if bit == 1 else 1
        return current_out, dp0, dp1
    else:
        child_size = size // 3
        # Compute dp for left child
        left_current, left_dp0, left_dp1 = compute_dp(low, child_size)
        # Compute dp for middle child
        mid_current, mid_dp0, mid_dp1 = compute_dp(low + child_size, child_size)
        # Compute dp for right child
        right_current, right_dp0, right_dp1 = compute_dp(low + 2 * child_size, child_size)
        
        # Compute current output of this node
        sum_curr = left_current + mid_current + right_current
        current_out = 1 if sum_curr >= 2 else 0
        
        # Compute dp1_min (min flips to make output 1)
        cost_all1 = left_dp1 + mid_dp1 + right_dp1
        cost_lm1_r0 = left_dp1 + mid_dp1 + right_dp0
        cost_lr1_m0 = left_dp1 + mid_dp0 + right_dp1
        cost_mr1_l0 = left_dp0 + mid_dp1 + right_dp1
        dp1_min = min(cost_all1, cost_lm1_r0, cost_lr1_m0, cost_mr1_l0)
        
        # Compute dp0_min (min flips to make output 0)
        cost_all0 = left_dp0 + mid_dp0 + right_dp0
        cost_lm0_r1 = left_dp0 + mid_dp0 + right_dp1
        cost_lr0_m1 = left_dp0 + mid_dp1 + right_dp0
        cost_mr0_l1 = left_dp1 + mid_dp0 + right_dp0
        dp0_min = min(cost_all0, cost_lm0_r1, cost_lr0_m1, cost_mr0_l1)
        
        return current_out, dp0_min, dp1_min

# Compute dp for the root
root_current, dp0_min, dp1_min = compute_dp(0, len(A))

# Minimum flips to change the output
if root_current == 0:
    ans = dp1_min
else:
    ans = dp0_min

# Output the answer
print(ans)