n, k = map(int, input().split())
s = input().strip()

# Find all 1-blocks
blocks = []
current_start = None
for i in range(len(s)):
    if s[i] == '1':
        if current_start is None:
            if i == 0 or s[i-1] == '0':
                current_start = i
    else:
        if current_start is not None:
            blocks.append((current_start, i - 1))
            current_start = None
# Add the last block if any
if current_start is not None:
    blocks.append((current_start, len(s) - 1))

# Process K
block_k_minus_1 = blocks[k-2]
block_k = blocks[k-1]

r_k_minus_1 = block_k_minus_1[1]
l_k, r_k = block_k[0], block_k[1]

inserted_block_length = r_k - l_k + 1
part1 = s[0 : r_k_minus_1 + 1]
part2 = '1' * inserted_block_length

start_insert = r_k_minus_1 + 1
end_insert = start_insert + inserted_block_length - 1
part3_start = end_insert + 1
part3_end = r_k

if part3_start <= part3_end:
    part3 = '0' * (part3_end - part3_start + 1)
else:
    part3 = ''

part4 = s[r_k + 1 :]

t = part1 + part2 + part3 + part4
print(t)