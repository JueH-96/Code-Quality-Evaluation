import sys
MOD = 998244353

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
K = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))

# Initialize DP
prefix_sum_map = {0: 1}  # Prefix sum to sum of f[j] for that prefix sum
total_f_sum = 1  # Sum of all f[j] computed so far
current_prefix = 0  # Current prefix sum value
ans = 0  # To store the final answer f[N]

# Iterate through each position from 1 to N
for i in range(1, N + 1):
    # Compute new prefix sum
    new_prefix = current_prefix + A[i - 1]
    
    # Compute target for bad sum query
    target = new_prefix - K
    
    # Get the bad sum from the map
    bad_sum = prefix_sum_map.get(target, 0)
    
    # Compute f_i with modulo operation and handle negative
    f_i = (total_f_sum - bad_sum) % MOD
    f_i = (f_i + MOD) % MOD
    
    # Store the current f_i (will be f[N] at the end)
    ans = f_i
    
    # Add f_i to the prefix sum map
    prefix_sum_map[new_prefix] = (prefix_sum_map.get(new_prefix, 0) + f_i) % MOD
    
    # Add f_i to the total sum
    total_f_sum = (total_f_sum + f_i) % MOD
    
    # Update current prefix sum for next iteration
    current_prefix = new_prefix

# Output the answer
print(ans)