n = int(input())
size = 3 ** n

for i in range(size):
    row = []
    for j in range(size):
        is_white = False
        for d in range(n):
            divisor = 3 ** d
            di = (i // divisor) % 3
            dj = (j // divisor) % 3
            if di == 1 and dj == 1:
                is_white = True
                break
        row.append('.' if is_white else '#')
    print(''.join(row))