n, k = map(int, input().split())
s = input().strip()

blocks = []
current_start = -1

for i in range(n):
    if s[i] == '1':
        if current_start == -1:
            current_start = i
    else:
        if current_start != -1:
            blocks.append((current_start, i - 1))
            current_start = -1
# Check if there's a block at the end
if current_start != -1:
    blocks.append((current_start, n - 1))

# Get the necessary blocks
prev_block = blocks[k-2]
current_block = blocks[k-1]
r_prev = prev_block[1]
l_k, r_k = current_block
len_k = r_k - l_k + 1

part1 = s[:r_prev + 1]
part2 = '1' * len_k
zeros_needed = r_k - r_prev - len_k
part3 = '0' * zeros_needed
part4 = s[r_k + 1:] if r_k + 1 < n else ''

result = part1 + part2 + part3 + part4
print(result)