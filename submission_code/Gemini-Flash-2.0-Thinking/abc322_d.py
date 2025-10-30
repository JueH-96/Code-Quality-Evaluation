class PolyominoSolver:
    def solve(self, polyomino_strs):
        self.polys = []
        for i in range(3):
            grid = [polyomino_strs[i * 4 + r] for r in range(4)]
            cells = set()
            for r in range(4):
                for c in range(4):
                    if grid[r][c] == '#':
                        cells.add((r, c))
            self.polys.append(self.normalize_global(cells))

        self.placements = [list(self.get_all_valid_placements(poly)) for poly in self.polys]

        all_grid_cells = frozenset((r, c) for r in range(4) for c in range(4))

        from itertools import product
        for p1_placement, p2_placement, p3_placement in product(self.placements[0], self.placements[1], self.placements[2]):
            cells1 = p1_placement
            cells2 = p2_placement
            cells3 = p3_placement

            overlap = cells1.intersection(cells2).union(cells1.intersection(cells3)).union(cells2.intersection(cells3))
            if overlap:
                continue

            covered = cells1.union(cells2).union(cells3)
            if covered == all_grid_cells:
                return "Yes"

        return "No"

    def normalize_global(self, cells):
        if not cells:
            return frozenset()
        min_r = min(r for r, c in cells)
        min_c = min(c for r, c in cells)
        return frozenset((r - min_r, c - min_c) for r, c in cells)

    def rotate_single(self, r, c):
        return -c, r

    def rotate(self, cells):
        return frozenset(self.rotate_single(c, r) for r, c in cells)

    def get_rotations(self, cells):
        rotations = set()
        normalized_cells = self.normalize_global(cells)
        current_rotation = normalized_cells
        for _ in range(4):
            rotations.add(current_rotation)
            rotated_cells = frozenset(self.rotate_single(c, r) for r, c in current_rotation)
            if rotated_cells:
                min_r = min(r for r, c in rotated_cells)
                min_c = min(c for r, c in rotated_cells)
                current_rotation = frozenset((r - min_r, c - min_c) for r, c in rotated_cells)
            else:
                current_rotation = frozenset()
        return set(frozenset(rc) for rc in rotations)

    def get_all_valid_placements(self, poly_cells):
        placements = set()
        normalized_poly = self.normalize_global(poly_cells)
        rotations = self.get_rotations(normalized_poly)

        for rotation in rotations:
            for r_offset in range(4):
                for c_offset in range(4):
                    placement = frozenset((r + r_offset, c + c_offset) for r, c in rotation)
                    if all(0 <= r < 4 and 0 <= c < 4 for r, c in placement):
                        placements.add(placement)
        return placements

def main():
    polyomino_strs = [input() for _ in range(12)]
    solver = PolyominoSolver()
    result = solver.solve(polyomino_strs)
    print(result)

if __name__ == "__main__":
    main()