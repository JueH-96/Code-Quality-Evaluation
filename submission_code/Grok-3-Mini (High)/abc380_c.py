import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
K = int(data[index])
index += 1
S = data[index]

# Find all 1-blocks with 0-based indices
blocks = []
i = 0
while i < N:
    if S[i] == '1':
        start_0 = i
        while i < N and S[i] == '1':
            i += 1
        end_0 = i - 1  # 0-based end index
        blocks.append((start_0, end_0))
    else:
        i += 1

# Get the (K-1)-th and K-th blocks (0-based indexing in list)
block_km1 = blocks[K-2]  # (start, end) for (K-1)-th block
block_k = blocks[K-1]    # (start, end) for K-th block

end_km1_0 = block_km1[1]  # end index 0-based of (K-1)-th block
len_K = block_k[1] - block_k[0] + 1  # length of K-th block
end_k_0 = block_k[1]  # end index 0-based of K-th block

# Create T as a list
T_list = [''] * N

# First part: copy S from index 0 to end_km1_0
for j in range(0, end_km1_0 + 1):
    T_list[j] = S[j]

# Second part: set '1' from index end_km1_0 + 1 to end_km1_0 + len_K
start_j2 = end_km1_0 + 1
end_j2 = end_km1_0 + len_K
for j in range(start_j2, end_j2 + 1):
    T_list[j] = '1'

# Zero part: set '0' from j_min_zero to j_max_zero
r_km1_1based = end_km1_0 + 1  # 1-based end of (K-1)-th block
j_min_zero = r_km1_1based + len_K
j_max_zero = end_k_0  # 0-based end of K-th block
for j in range(j_min_zero, j_max_zero + 1):
    T_list[j] = '0'

# Last part: copy S from index end_k_0 + 1 to N-1
start_last = end_k_0 + 1
for j in range(start_last, N):
    T_list[j] = S[j]

# Join and print the result
result = ''.join(T_list)
print(result)