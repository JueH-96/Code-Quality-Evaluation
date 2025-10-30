n = int(input())
size = 3 ** n

def is_white(n, i, j):
    for k in range(1, n + 1):
        block_size = 3 ** (k - 1)
        x = i // block_size
        y = j // block_size
        if x % 3 == 1 and y % 3 == 1:
            return True
    return False

for i in range(size):
    row = []
    for j in range(size):
        if is_white(n, i, j):
            row.append('.')
        else:
            row.append('#')
    print(''.join(row))