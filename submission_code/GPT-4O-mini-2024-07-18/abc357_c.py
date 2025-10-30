def generate_carpet(n):
    size = 3 ** n
    carpet = [['#'] * size for _ in range(size)]

    def fill_carpet(x, y, level):
        if level == 0:
            return
        block_size = 3 ** (level - 1)
        # Fill the 8 blocks
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:  # Central block
                    for a in range(block_size):
                        for b in range(block_size):
                            carpet[x + i * block_size + a][y + j * block_size + b] = '.'
                else:
                    fill_carpet(x + i * block_size, y + j * block_size, level - 1)

    fill_carpet(0, 0, n)
    return carpet

def print_carpet(carpet):
    for row in carpet:
        print(''.join(row))

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    carpet = generate_carpet(N)
    print_carpet(carpet)