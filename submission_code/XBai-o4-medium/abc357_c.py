n = int(input())

def is_black(K, i, j):
    if K == 0:
        return True
    block_size = 3 ** (K - 1)
    row = i // block_size
    col = j // block_size
    if row == 1 and col == 1:
        return False
    else:
        return is_black(K - 1, i % block_size, j % block_size)

size = 3 ** n

for i in range(size):
    line = ''
    for j in range(size):
        line += '#' if is_black(n, i, j) else '.'
    print(line)