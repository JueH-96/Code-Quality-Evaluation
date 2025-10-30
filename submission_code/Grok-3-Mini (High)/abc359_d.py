import sys

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
K = int(data[index + 1])
index += 2
S = data[index]

MOD = 998244353
M = K - 1
num_masks = 1 << M

# Initialize dp_prev
dp_prev = [0] * num_masks
dp_prev[0] = 1

# Iterate through each position
for i in range(1, N + 1):
    dp_curr = [0] * num_masks
    char_pos = S[i - 1]
    if char_pos == 'A':
        poss_c = [0]
    elif char_pos == 'B':
        poss_c = [1]
    elif char_pos == '?':
        poss_c = [0, 1]
    else:
        # Invalid character, but should not happen
        poss_c = []
    
    # Iterate over all previous masks
    for mask_prev in range(num_masks):
        if dp_prev[mask_prev] == 0:
            continue  # Skip if no ways to reach this state
        for c in poss_c:
            if i >= K:
                # Check if the substring ending at i is a palindrome
                palindrome = True
                len_j_max = (K - 1) // 2
                for j in range(len_j_max + 1):
                    left_val = (mask_prev >> j) & 1
                    if j == 0:
                        right_val = c
                    else:
                        right_val = (mask_prev >> (K - 1 - j)) & 1
                    if left_val != right_val:
                        palindrome = False
                        break
                if palindrome:
                    continue  # Skip if palindrome
            # Not a palindrome or i < K, proceed
            # Compute new mask
            new_mask = (mask_prev >> 1) | (c << (M - 1))
            # Add to dp_curr with modulo
            dp_curr[new_mask] += dp_prev[mask_prev]
            dp_curr[new_mask] %= MOD
    
    # Set dp_prev to dp_curr for next iteration
    dp_prev = dp_curr

# Sum all values in dp_prev and take modulo
ans = 0
for val in dp_prev:
    ans += val
    ans %= MOD

# Output the answer
print(ans)