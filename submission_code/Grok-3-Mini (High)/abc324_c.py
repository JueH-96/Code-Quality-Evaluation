import sys
import bisect

# Read all input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
T_prime = data[index]
index += 1

# Read N strings into a list
S_list = []
for i in range(N):
    S = data[index]
    index += 1
    S_list.append(S)

# Length of T'
len_T = len(T_prime)

# Preprocess position lists for each character in T'
pos = [[] for _ in range(26)]
for j in range(len_T):
    char_idx = ord(T_prime[j]) - ord('a')
    pos[char_idx].append(j)

# List to store the 1-based indices that satisfy the condition
result = []

# Iterate through each S_i with its index
for idx in range(N):  # idx is 0-based index in S_list
    S = S_list[idx]
    len_S = len(S)
    delta = len_S - len_T
    
    if abs(delta) > 1:
        continue  # Difference in length more than 1, impossible
    elif delta == 0:
        # Same length, check if equal or differ by exactly one character
        diff_count = 0
        for p in range(len_T):
            if S[p] != T_prime[p]:
                diff_count += 1
                if diff_count > 1:
                    break  # Early termination if more than one difference
        if diff_count <= 1:
            result.append(idx + 1)  # 1-based index
    elif delta == -1:
        # len_S = len_T - 1, check if S is a subsequence of T'
        last_pos = -1
        matched = True
        for char in S:
            char_idx = ord(char) - ord('a')
            pos_list = pos[char_idx]
            # Binary search to find smallest position > last_pos
            idx_in_list = bisect.bisect_left(pos_list, last_pos + 1)
            if idx_in_list == len(pos_list):
                matched = False
                break
            else:
                last_pos = pos_list[idx_in_list]
        if matched:
            result.append(idx + 1)  # 1-based index
    elif delta == 1:
        # len_S = len_T + 1, check if T' is a subsequence of S
        t_idx = 0
        for char_s in S:
            if t_idx < len_T and char_s == T_prime[t_idx]:
                t_idx += 1
        if t_idx == len_T:
            result.append(idx + 1)  # 1-based index

# Output the result
K = len(result)
print(K)
if K > 0:
    print(' '.join(map(str, result)))