def generate_carpet(N):
    size = 3 ** N
    carpet = [['#' for _ in range(size)] for _ in range(size)]

    def fill_carpet(x, y, level):
        if level == 0:
            return
        block_size = 3 ** (level - 1)
        for i in range(block_size):
            for j in range(block_size):
                carpet[x + block_size + i][y + block_size + j] = '.'
        for dx in range(3):
            for dy in range(3):
                if dx == 1 and dy == 1:
                    continue
                fill_carpet(x + dx * block_size, y + dy * block_size, level - 1)

    fill_carpet(0, 0, N)
    for row in carpet:
        print(''.join(row))

import sys
input = sys.stdin.read
N = int(input().strip())
generate_carpet(N)