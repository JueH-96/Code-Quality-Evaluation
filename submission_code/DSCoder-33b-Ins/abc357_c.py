# YOUR CODE HERE

def print_carpet(n, x, y):
    if n == 0:
        carpet[x][y] = '#'
    else:
        unit = 3 ** (n - 1)
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                print_carpet(n - 1, x + i * unit, y + j * unit)

N = int(input())
size = 3 ** N
carpet = [['.' for _ in range(size)] for _ in range(size)]
print_carpet(N, 0, 0)

for row in carpet:
    print(''.join(row))