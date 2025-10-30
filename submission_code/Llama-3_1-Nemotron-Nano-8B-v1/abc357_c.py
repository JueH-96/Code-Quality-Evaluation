n = int(input())
size = 3 ** n

for i in range(size):
    row = []
    for j in range(size):
        is_white = False
        for m in range(1, n + 1):
            s_m = 3 ** m
            block_size = 3 ** (m - 1)
            start_i = (s_m - block_size) // 2
            start_j = start_i
            if (start_i <= i < start_i + block_size and
                start_j <= j < start_j + block_size):
                is_white = True
                break
        row.append('.' if is_white else '#')
    print(''.join(row))