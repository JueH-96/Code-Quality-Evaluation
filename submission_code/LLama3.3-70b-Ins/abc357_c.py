def draw_carpet(n, size, x, y):
    if n == 0:
        return

    sub_size = size // 3

    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                for k in range(sub_size):
                    for m in range(sub_size):
                        carpet[y + k][x + m] = '.'
            else:
                draw_carpet(n - 1, sub_size, x + j * sub_size, y + i * sub_size)


def main():
    n = int(input())
    size = 3 ** n
    global carpet
    carpet = [['#' for _ in range(size)] for _ in range(size)]

    draw_carpet(n, size, 0, 0)

    for row in carpet:
        print(''.join(row))


if __name__ == "__main__":
    main()