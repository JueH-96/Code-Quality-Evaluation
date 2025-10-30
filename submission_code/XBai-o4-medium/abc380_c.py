n, k = map(int, input().split())
s = input().strip()

blocks = []
i = 0
while i < n:
    if s[i] == '1':
        start = i
        while i < n and s[i] == '1':
            i += 1
        end = i - 1
        blocks.append((start, end))
    else:
        i += 1

prev_block = blocks[k-2]
current_block = blocks[k-1]

r_prev = prev_block[1]
l_k = current_block[0]
r_k = current_block[1]

len_k = r_k - l_k + 1
zeros_count = l_k - r_prev - 1

new_s = s[:r_prev+1] + '1' * len_k + '0' * zeros_count + s[r_k+1:]

print(new_s)