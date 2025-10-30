import sys

# Read input
N, K = map(int, sys.stdin.readline().split())
S = sys.stdin.readline().strip()

# Find all 1-blocks and store their 0-based start and end indices
blocks = []
current_start = None
for i in range(N):
    if S[i] == '1':
        # If we just started a '1' sequence
        if current_start is None:
            current_start = i
        # Check if the '1' sequence ends at this position
        # It ends if it's the last character or the next character is '0'
        if i + 1 == N or S[i+1] == '0':
            blocks.append((current_start, i))
            current_start = None
    # If S[i] is '0', and current_start is not None, it means the previous
    # character was '1' and the block ended there. The check `i + 1 == N or S[i+1] == '0'`
    # inside the 'S[i] == '1'' block correctly handles this transition.

# Get indices of the (K-1)-th and K-th blocks (0-based)
# K is 1-based in problem, list index is K-1.
# (K-1)-th block (1-based) is index K-2 in list.
# K-th block (1-based) is index K-1 in list.
# Problem guarantees S contains at least K 1-blocks, and K >= 2.
# So, blocks list has length >= K, and indices K-2 and K-1 are valid.
l_k_minus_1, r_k_minus_1 = blocks[K-2]
l_k, r_k = blocks[K-1]

# Calculate lengths
len_k_block = r_k - l_k + 1
# Length of the zero gap originally between (K-1)-th and K-th blocks
# Indices in S are r_k_minus_1 + 1, ..., l_k - 1
# The number of indices is (l_k - 1) - (r_k_minus_1 + 1) + 1 = l_k - r_k_minus_1 - 1
# This length will be 0 if the blocks are adjacent (e.g., "111" + "11")
len_zero_gap = l_k - r_k_minus_1 - 1 # This length is guaranteed >= 0

# Construct the resulting string T based on the rules provided:
# The problem description defines T by specifying characters within index ranges (1-based):
# T_i = S_i for 1 <= i <= r_{K-1}
# T_i = 1 for r_{K-1} + 1 <= i <= r_{K-1} + (r_K - l_K) + 1
# T_i = 0 for r_{K-1} + (r_K - l_K) + 2 <= i <= r_K
# T_i = S_i for r_K + 1 <= i <= N

# Translating to 0-based indices and Python slicing:
# T[0 .. r'_{K-1}] = S[0 .. r'_{K-1}]                  -> S[:r_k_minus_1 + 1]
# T[r'_{K-1}+1 .. r'_{K-1} + (r'_K - l'_K) + 1] = '1' * len_k_block
# T[r'_{K-1} + (r'_K - l'_K) + 2 .. r'_K] = '0' * len_zero_gap
# T[r'_K + 1 .. N-1] = S[r'_K + 1 .. N-1]              -> S[r_k + 1:]

# These segments cover the entire new string T in order:
# 1. The part of S up to the end of the (K-1)-th block.
# 2. The K-th block itself (moved).
# 3. The sequence of '0's that was originally between the (K-1)-th and K-th blocks.
# 4. The part of S that was originally after the K-th block.

prefix = S[:r_k_minus_1 + 1]
moved_block_str = '1' * len_k_block
zero_gap_str = '0' * len_zero_gap
suffix = S[r_k + 1:] # Python slice S[start:] goes up to the end of the string (N-1)

T = prefix + moved_block_str + zero_gap_str + suffix

# Print the result
print(T)