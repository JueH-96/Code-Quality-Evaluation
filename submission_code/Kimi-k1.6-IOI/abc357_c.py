n = int(input())
size = 3 ** n
for i in range(size):
    row = []
    for j in range(size):
        white = False
        for k in range(n):
            i_k = (i // (3 ** k)) % 3
            j_k = (j // (3 ** k)) % 3
            if i_k == 1 and j_k == 1:
                white = True
                break
        row.append('.' if white else '#')
    print(''.join(row))