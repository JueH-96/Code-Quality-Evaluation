n, K = map(int, input().split())
S = input().strip()

blocks = []
i = 0
n_s = len(S)
while i < n_s:
    if S[i] == '1':
        start = i
        while i < n_s and S[i] == '1':
            i += 1
        end = i - 1
        blocks.append((start, end))
    else:
        i += 1

prev_block = blocks[K-2]
current_block = blocks[K-1]

prev_block_end = prev_block[1]
current_block_start, current_block_end = current_block
current_block_length = current_block_end - current_block_start + 1

part1 = S[: prev_block_end + 1]
part2 = '1' * current_block_length
part3 = '0' * (current_block_end - (prev_block_end + current_block_length))
part4 = S[current_block_end + 1:]

result = part1 + part2 + part3 + part4
print(result)