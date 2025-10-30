def solve():
    grid = []
    for _ in range(3):
        grid.append(list(map(int, input().split())))

    lines = [
        [(0, 0), (0, 1), (0, 2)],
        [(1, 0), (1, 1), (1, 2)],
        [(2, 0), (2, 1), (2, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 1), (1, 1), (2, 1)],
        [(0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 1), (2, 2)],
        [(2, 0), (1, 1), (0, 2)],
    ]

    from itertools import permutations

    all_cells = [(i, j) for i in range(3) for j in range(3)]
    num_good_permutations = 0
    total_permutations = 0

    for perm in permutations(all_cells):
        total_permutations += 1
        is_disappointed = False
        for line in lines:
            c1, c2, c3 = line
            v1 = grid[c1[0]][c1[1]]
            v2 = grid[c2[0]][c2[1]]
            v3 = grid[c3[0]][c3[1]]

            p1 = perm.index(c1)
            p2 = perm.index(c2)
            p3 = perm.index(c3)

            ordered_indices = sorted([(p1, c1), (p2, c2), (p3, c3)])

            first_seen_cell = ordered_indices[0][1]
            second_seen_cell = ordered_indices[1][1]
            third_seen_cell = ordered_indices[2][1]

            val_first = grid[first_seen_cell[0]][first_seen_cell[1]]
            val_second = grid[second_seen_cell[0]][second_seen_cell[1]]
            val_third = grid[third_seen_cell[0]][third_seen_cell[1]]

            if val_first == val_second and val_first != val_third:
                is_disappointed = True
                break

        if not is_disappointed:
            num_good_permutations += 1

    probability = num_good_permutations / total_permutations
    print(probability)

solve()