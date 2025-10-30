def print_carpet(N):
    if N == 0:
        print("#")
        return

    size = 3 ** N
    carpet = [["." for _ in range(size)] for _ in range(size)]

    def build_carpet(k, x, y):
        if k == 0:
            carpet[x][y] = "#"
            return

        block_size = 3 ** (k - 1)
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                build_carpet(k - 1, x + i * block_size, y + j * block_size)

    build_carpet(N, 0, 0)

    for row in carpet:
        print("".join(row))

# Read input from stdin
N = int(input())

# Print the carpet
print_carpet(N)