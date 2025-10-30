def is_black(i, j, N):
    k = N
    while k > 0:
        s = 3 ** k
        block_size = s // 3
        x_block = i // block_size
        y_block = j // block_size
        if x_block == 1 and y_block == 1:
            return False
        i %= block_size
        j %= block_size
        k -= 1
    return True

N = int(input())
size = 3 ** N
for i in range(size):
    row = []
    for j in range(size):
        if is_black(i, j, N):
            row.append('#')
        else:
            row.append('.')
    print(''.join(row))