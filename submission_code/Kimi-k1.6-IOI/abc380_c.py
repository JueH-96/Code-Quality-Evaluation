n, k = map(int, input().split())
s = input().strip()

blocks = []
i = 0
while i < len(s):
    if s[i] == '1':
        start = i
        while i < len(s) and s[i] == '1':
            i += 1
        end = i - 1
        blocks.append((start, end))
    else:
        i += 1

k_idx = k - 1
prev_block = blocks[k_idx - 1]
current_block = blocks[k_idx]

prev_r = prev_block[1]
original_l, original_r = current_block
original_len = original_r - original_l + 1

new_start = prev_r + 1
new_end = new_start + original_len - 1
zero_start = new_end + 1
zero_end = original_r

part1 = s[0:prev_r + 1]
part2 = '1' * original_len
num_zeros = max(0, zero_end - zero_start + 1)
part3 = '0' * num_zeros
part4 = s[original_r + 1:]

result = part1 + part2 + part3 + part4
print(result)