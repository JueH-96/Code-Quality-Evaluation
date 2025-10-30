import sys
from itertools import product, combinations, permutations

def main():
    n = int(sys.stdin.readline())
    r = sys.stdin.readline().strip()
    c = sys.stdin.readline().strip()

    # Generate all possible j_i combinations
    for j_comb in product(range(n), repeat=n):
        grid = [['.'] * n for _ in range(n)]
        for i in range(n):
            j_i = j_comb[i]
            grid[i][j_i] = r[i]

        # Check columns' first non-dot
        valid = True
        for j in range(n):
            first = None
            for i in range(n):
                if grid[i][j] != '.':
                    first = grid[i][j]
                    break
            if first is None or first != c[j]:
                valid = False
                break
        if not valid:
            continue

        # Check available positions for each row
        row_info = []
        valid_row = True
        for i in range(n):
            j_i = j_comb[i]
            available = []
            for j in range(j_i + 1, n):
                available.append(j)
            if len(available) < 2:
                valid_row = False
                break
            other_letters = [let for let in ['A', 'B', 'C'] if let != r[i]]
            row_info.append((available, other_letters))
        if not valid_row:
            continue

        # Generate row options
        row_options = []
        for i in range(n):
            available, letters = row_info[i]
            pos_pairs = list(combinations(available, 2))
            opts = []
            for pos in pos_pairs:
                for pl in permutations(letters):
                    opts.append((pos[0], pos[1], pl[0], pl[1]))
            if not opts:
                valid_row = False
                break
            row_options.append(opts)
        if not valid_row:
            continue

        # Try all possible row assignments
        for row_assign in product(*row_options):
            new_grid = [row.copy() for row in grid]
            for i in range(n):
                pos0, pos1, let0, let1 = row_assign[i]
                new_grid[i][pos0] = let0
                new_grid[i][pos1] = let1

            # Check columns
            valid_cols = True
            for j in range(n):
                col_chars = []
                for i in range(n):
                    if new_grid[i][j] != '.':
                        col_chars.append(new_grid[i][j])
                if sorted(col_chars) != ['A', 'B', 'C']:
                    valid_cols = False
                    break
            if not valid_cols:
                continue

            # Check rows
            valid_rows = True
            for i in range(n):
                row_chars = []
                for ch in new_grid[i]:
                    if ch != '.':
                        row_chars.append(ch)
                if sorted(row_chars) != ['A', 'B', 'C']:
                    valid_rows = False
                    break
            if valid_rows:
                print("Yes")
                for row in new_grid:
                    print(''.join(row))
                return

    print("No")

if __name__ == "__main__":
    main()