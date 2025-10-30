import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index+1])
index += 2
S_str = data[index]
T_str = data[index+1]
S = [int(c) for c in S_str]
T = [int(c) for c in T_str]

# Precompute suffix max for T
suffix_max = [0] * M
if M > 1:
    for i in range(M-2, -1, -1):
        suffix_max[i] = max(T[i+1], suffix_max[i+1])

for k in range(M):
    D = T[k]
    max_f = suffix_max[k]
    # Find leftmost i with S[i] < D
    left_i = -1
    for i in range(N):
        if S[i] < D:
            left_i = i
            break
    if left_i != -1:
        if D > max_f:
            i = left_i  # leftmost
        else:
            # Find rightmost i with S[i] < D
            right_i = -1
            for j in range(N-1, -1, -1):
                if S[j] < D:
                    right_i = j
                    break
            i = right_i
    else:
        # No position with S[i] < D, place on rightmost position
        i = N-1
    # Replace S[i] with D
    S[i] = D

# Output the result
ans = ''.join(str(d) for d in S)
print(ans)