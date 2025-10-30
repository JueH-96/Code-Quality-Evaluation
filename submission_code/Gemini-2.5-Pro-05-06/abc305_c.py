import sys

def solve():
    H, W = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        # .strip() is important to remove trailing newline characters
        grid.append(sys.stdin.readline().strip())

    # Initialize coordinates for the bounding box of '#' characters.
    # min_r_hash will store the smallest row index of a '#'.
    # max_r_hash will store the largest row index of a '#'.
    # Similar for min_c_hash and max_c_hash for column indices.
    # Initial values are set such that any valid index will update them.
    min_r_hash = H 
    max_r_hash = -1
    min_c_hash = W
    max_c_hash = -1

    # Find the bounding box of all '#' characters.
    # The problem states that originally cookies formed a rectangle
    # whose height and width were at least 2.
    # Snuke ate one cookie from this rectangle.
    # Because height and width are at least 2, removing one cookie from any
    # boundary row/column of the rectangle (e.g., the topmost row, or leftmost column)
    # will not cause that entire row/column (within the rectangle's extent) 
    # to become empty of cookies.
    # Thus, the minimum/maximum row/column of the remaining '#' characters
    # will still correspond to the boundaries of the original rectangle.
    for r_idx in range(H):
        for c_idx in range(W):
            if grid[r_idx][c_idx] == '#':
                min_r_hash = min(min_r_hash, r_idx)
                max_r_hash = max(max_r_hash, r_idx)
                min_c_hash = min(min_c_hash, c_idx)
                max_c_hash = max(max_c_hash, c_idx)
    
    # The problem guarantees that there was an initial rectangle of cookies,
    # and Snuke ate one. Since the original rectangle height/width >= 2,
    # it had at least 2*2=4 cookies. After one is eaten, at least 3 cookies remain.
    # So, min_r_hash, max_r_hash, etc., will be correctly updated from their initial values.

    # Iterate within this determined bounding box (which represents the original rectangle).
    # The eaten cookie corresponds to the unique cell (r_idx, c_idx) within this box
    # such that grid[r_idx][c_idx] is '.'.
    for r_idx in range(min_r_hash, max_r_hash + 1):
        for c_idx in range(min_c_hash, max_c_hash + 1):
            if grid[r_idx][c_idx] == '.':
                # Output 1-indexed coordinates.
                # str() conversion and concatenation are used for sys.stdout.write.
                sys.stdout.write(str(r_idx + 1) + " " + str(c_idx + 1) + "
")
                # Found the unique eaten cookie, so we can exit.
                return

if __name__ == '__main__':
    solve()