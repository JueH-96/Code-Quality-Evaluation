n = int(input())

def generate(k):
    if k == 0:
        return [['#']]
    else:
        sub_size = 3 ** (k - 1)
        subcarpets = []
        for i in range(3):
            row = []
            for j in range(3):
                if i == 1 and j == 1:
                    # Center subgrid: all '.'.
                    subgrid = [['.'] * sub_size for _ in range(sub_size)]
                else:
                    subgrid = generate(k - 1)
                row.extend(subgrid)
            subcarpets.append(row)
        main_grid = []
        for i in range(3):
            main_row = []
            for j in range(3):
                main_row.extend(subcarpets[i][j])
            main_grid.append(main_row)
        return main_grid

carpet = generate(n)
for row in carpet:
    print(''.join(row))