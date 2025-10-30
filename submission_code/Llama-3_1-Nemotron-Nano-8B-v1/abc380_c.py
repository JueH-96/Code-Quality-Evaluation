N, K = map(int, input().split())
S = input().strip()

blocks = []
current_block_start = None

for i in range(N):
    if S[i] == '1':
        if i == 0 or S[i-1] == '0':
            current_block_start = i
    else:
        if current_block_start is not None:
            blocks.append((current_block_start + 1, i))
            current_block_start = None
if current_block_start is not None:
    blocks.append((current_block_start + 1, N))

block_prev = blocks[K-2]
block_k = blocks[K-1]

r_prev = block_prev[1]
l_k, r_k = block_k
len_k = r_k - l_k + 1
r_prev_0 = r_prev - 1
r_k_0 = r_k - 1

part1 = S[:r_prev]
part2 = '1' * len_k
part3_length = r_k_0 - (r_prev_0 + len_k)
part3 = '0' * part3_length
part4 = S[r_k_0 + 1:]

print(part1 + part2 + part3 + part4)