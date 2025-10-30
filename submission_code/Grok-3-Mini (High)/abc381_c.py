import sys
data = sys.stdin.read().split()
N = int(data[0])
S = data[1]

# Compute len_end_1: length of consecutive '1's ending at index i
len_end_1 = [0] * N
for i in range(N):
    if S[i] == '1':
        if i == 0 or S[i-1] != '1':
            len_end_1[i] = 1
        else:
            len_end_1[i] = len_end_1[i-1] + 1
    else:
        len_end_1[i] = 0

# Compute len_start_2: length of consecutive '2's starting at index i
len_start_2 = [0] * N
if N > 0:
    if S[N-1] == '2':
        len_start_2[N-1] = 1
    for i in range(N-2, -1, -1):
        if S[i] == '2':
            if S[i+1] == '2':
                len_start_2[i] = len_start_2[i+1] + 1
            else:
                len_start_2[i] = 1
        else:
            len_start_2[i] = 0  # Ensure it's 0, though already initialized

# Find the maximum length of 11/22 substring
ans = 0
for m in range(N):
    if S[m] == '/':
        # Compute max K for left part
        if m - 1 >= 0 and S[m - 1] == '1':
            max_K_left = len_end_1[m - 1]
        else:
            max_K_left = 0
        # Compute max K for right part
        if m + 1 < N and S[m + 1] == '2':
            max_K_right = len_start_2[m + 1]
        else:
            max_K_right = 0
        # K is limited by the minimum of left and right
        K_min = min(max_K_left, max_K_right)
        # Length of the substring
        substr_len = 2 * K_min + 1
        # Update answer
        if substr_len > ans:
            ans = substr_len

# Output the result
print(ans)