def solve():
    n, h, w = map(int, input().split())
    dims = [tuple(map(int, input().split())) for _ in range(n)]

    from itertools import combinations

    for i in range(1 << n):
        subset_indices = [j for j in range(n) if (i >> j) & 1]
        subset_dims_orig = [dims[j] for j in subset_indices]

        total_area = sum(dim[0] * dim[1] for dim in subset_dims_orig)

        if total_area == h * w:
            from itertools import permutations

            def can_place(grid, tiles):
                if not tiles:
                    return all(all(cell > 0 for cell in row) for row in grid)

                r, c = -1, -1
                for i_r in range(h):
                    for i_c in range(w):
                        if grid[i_r][i_c] == 0:
                            r, c = i_r, i_c
                            break
                    if r != -1:
                        break

                if r == -1:
                    return False

                current_tile_dim = tiles[0]
                remaining_tiles = tiles[1:]

                h1, w1 = current_tile_dim
                h2, w2 = current_tile_dim[1], current_tile_dim[0]

                # Try orientation 1
                if r + h1 <= h and c + w1 <= w:
                    can_fit = True
                    for i_r in range(h1):
                        for i_c in range(w1):
                            if grid[r + i_r][c + i_c] > 0:
                                can_fit = False
                                break
                        if not can_fit:
                            break
                    if can_fit:
                        new_grid = [row[:] for row in grid]
                        for i_r in range(h1):
                            for i_c in range(w1):
                                new_grid[r + i_r][c + i_c] = 1
                        if can_place(new_grid, remaining_tiles):
                            return True

                # Try orientation 2
                if r + h2 <= h and c + w2 <= w:
                    can_fit = True
                    for i_r in range(h2):
                        for i_c in range(w2):
                            if grid[r + i_r][c + i_c] > 0:
                                can_fit = False
                                break
                        if not can_fit:
                            break
                    if can_fit:
                        new_grid = [row[:] for row in grid]
                        for i_r in range(h2):
                            for i_c in range(w2):
                                new_grid[r + i_r][c + i_c] = 1
                        if can_place(new_grid, remaining_tiles):
                            return True

                return False

            from itertools import product
            possible_orientations = []
            for dims_orig in subset_dims_orig:
                possible_orientations.append([(dims_orig[0], dims_orig[1]), (dims_orig[1], dims_orig[0])])

            from itertools import product

            num_subset_tiles = len(subset_dims_orig)

            def find_placements(index, current_grid):
                if index == num_subset_tiles:
                    return all(all(cell > 0 for cell in row) for row in current_grid)

                r, c = -1, -1
                for i_r in range(h):
                    for i_c in range(w):
                        if current_grid[i_r][i_c] == 0:
                            r, c = i_r, i_c
                            break
                    if r != -1:
                        break

                if r == -1:
                    return False

                original_dim = subset_dims_orig[index]
                dims_to_try = [(original_dim[0], original_dim[1]), (original_dim[1], original_dim[0])]

                for tile_h, tile_w in dims_to_try:
                    if r + tile_h <= h and c + tile_w <= w:
                        can_fit = True
                        for i_r in range(tile_h):
                            for i_c in range(tile_w):
                                if current_grid[r + i_r][c + i_c] > 0:
                                    can_fit = False
                                    break
                            if not can_fit:
                                break
                        if can_fit:
                            new_grid = [row[:] for row in current_grid]
                            for i_r in range(tile_h):
                                for i_c in range(tile_w):
                                    new_grid[r + i_r][c + i_c] = 1
                            if find_placements(index + 1, new_grid):
                                return True
                return False

            if find_placements(0, [[0 for _ in range(w)] for _ in range(h)]):
                print("Yes")
                return

    print("No")

solve()