n, k = map(int, input().split())
s = input().strip()

def find_blocks(s):
    blocks = []
    current_start = -1
    for i in range(len(s)):
        if s[i] == '1':
            if current_start == -1:
                current_start = i
        else:
            if current_start != -1:
                blocks.append((current_start, i - 1))
                current_start = -1
    # After loop, check if there's a remaining block
    if current_start != -1:
        blocks.append((current_start, len(s) - 1))
    return blocks

blocks = find_blocks(s)
if len(blocks) < k:
    print(s)
else:
    prev_block = blocks[k-2]
    curr_block = blocks[k-1]
    len_bk = curr_block[1] - curr_block[0] + 1
    zero_len = curr_block[1] - prev_block[1] - len_bk
    left_part = s[:prev_block[1]+1]
    middle_part = '1' * len_bk
    zero_part = '0' * zero_len
    right_part = s[curr_block[1]+1:]
    new_s = left_part + middle_part + zero_part + right_part
    print(new_s)