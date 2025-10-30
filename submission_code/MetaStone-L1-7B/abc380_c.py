n, k = map(int, input().split())
s = input().strip()

# Parse the string into blocks
blocks = []
prev_char = s[0]
start = 0
for i in range(1, len(s)):
    if s[i] != prev_char:
        end = i - 1
        blocks.append((start, end, prev_char))
        start = i
        prev_char = s[i]
end = len(s) - 1
blocks.append((start, end, prev_char))

# Extract list of 1-blocks
list_1_blocks = [b for b in blocks if b[2] == '1']

# Find the indices of the (K-2)-th and (K-1)-th 1-blocks
prev_idx = list_1_blocks.index(list_1_blocks[k-2])
target_idx = list_1_blocks.index(list_1_blocks[k-1])

# Construct the new block list
new_blocks = blocks[:prev_idx] + [list_1_blocks[k-2]] + [list_1_blocks[k-1]] + blocks[target_idx+1:]

# Construct the resulting string
result = []
for b in new_blocks:
    result.append(s[b[0]:b[1]+1])

print(''.join(result))