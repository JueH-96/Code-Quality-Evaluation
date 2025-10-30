n = int(input())
size = 3 ** n
for i in range(size):
    row = []
    for j in range(size):
        white = False
        for d in range(n):
            divisor = 3 ** d
            i_d = (i // divisor) % 3
            j_d = (j // divisor) % 3
            if i_d == 1 and j_d == 1:
                white = True
                break
        row.append('.' if white else '#')
    print(''.join(row))