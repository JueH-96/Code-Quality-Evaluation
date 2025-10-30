def solve():
    import sys
    sys.setrecursionlimit(10**7)

    # Read inputs
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    H = int(data[1])
    W = int(data[2])
    tiles = []
    idx = 3
    for _ in range(N):
        A = int(data[idx])
        B = int(data[idx+1])
        idx += 2
        tiles.append((A, B))

    # Total area we need to fill
    total_area = H * W

    # Quick check: if no tile can possibly cover exactly total_area in sum of subsets, answer is No
    # But first let's gather all subsets whose sum of areas = total_area
    # There are at most 2^N subsets (N <= 7), so we can afford it
    from itertools import combinations

    tile_areas = [a*b for (a,b) in tiles]
    candidate_subsets = []  # list of lists of tile indices
    # Generate all subsets of the set {0,1,...N-1}
    for subset_size in range(N+1):
        for subset_indices in combinations(range(N), subset_size):
            s = sum(tile_areas[i] for i in subset_indices)
            if s == total_area:
                candidate_subsets.append(list(subset_indices))

    # If no subset has the correct total area, print No
    if not candidate_subsets:
        print("No")
        return

    # We'll do a standard backtracking approach to see if we can tile the HxW grid
    # with a chosen subset of tiles. Each tile may be oriented (A_i x B_i) or (B_i x A_i).
    # We must fill the entire grid exactly once. We'll place tiles one by one, always
    # picking the top-leftmost empty cell next.
    # Because each tile is distinct, we must keep track of which have been used from the subset.

    # Prepare a grid to mark usage
    # We will define a recursive function and try each tile that is not used yet.
    # If it fits, we place it and recurse.

    def backtrack_fill(grid, h, w, subset_tiles, used):
        # Find the next empty cell in row-major order
        # If none found, we have a fully covered grid -> success
        best_row, best_col = -1, -1
        for rr in range(h):
            for cc in range(w):
                if not grid[rr][cc]:
                    best_row, best_col = rr, cc
                    break
            if best_row != -1:
                break
        if best_row == -1:
            # No empty cell -> success
            return True

        # Try each tile that is not used yet
        for i in range(len(subset_tiles)):
            if not used[i]:
                A, B = subset_tiles[i]
                # Try both orientations
                for (rA, rB) in [(A,B), (B,A)]:
                    # Check if it can fit in the grid at (best_row, best_col)
                    # and also not colliding with used cells
                    if best_row + rA <= h and best_col + rB <= w:
                        # Verify collision
                        can_place = True
                        for rr in range(best_row, best_row + rA):
                            for cc in range(best_col, best_col + rB):
                                if grid[rr][cc]:
                                    can_place = False
                                    break
                            if not can_place:
                                break
                        if can_place:
                            # Place it
                            for rr in range(best_row, best_row + rA):
                                for cc in range(best_col, best_col + rB):
                                    grid[rr][cc] = True
                            used[i] = True

                            # Recurse
                            if backtrack_fill(grid, h, w, subset_tiles, used):
                                return True

                            # Un-place
                            used[i] = False
                            for rr in range(best_row, best_row + rA):
                                for cc in range(best_col, best_col + rB):
                                    grid[rr][cc] = False
        return False

    # Try each candidate subset. If any works, print Yes
    for subset_indices in candidate_subsets:
        # Build a list of the tile dimensions for that subset
        subset_tiles = [tiles[i] for i in subset_indices]
        used = [False]*len(subset_tiles)
        # Reset the grid
        grid = [[False]*W for _ in range(H)]
        # Try to fill
        if backtrack_fill(grid, H, W, subset_tiles, used):
            print("Yes")
            return

    # If nothing worked
    print("No")

def main():
    solve()

if __name__ == "__main__":
    main()