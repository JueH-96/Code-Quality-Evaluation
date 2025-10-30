def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    grid = data[1:]
    # grid[i] is a string of length N (each character is '.' or '#')

    # We will construct the final arrangement in O(N^2) by observing that:
    # Each cell (r, c) (0-based) belongs to all "layers" i with i <= min(r, c, N-1-r, N-1-c).
    # In 0-based indexing, those layers i go from 0 up to that minimum.
    # Hence the cell (r, c) is rotated exactly (rI+1) times, where
    #   rI = min(r, c, N-1-r, N-1-c).
    #
    # A single 90-degree clockwise rotation in 0-based coordinates sends (r, c) -> (c, N-1-r).
    # Repeating it k times is equivalent to k mod 4 rotations.  Let k = (rI+1).
    # Then final_position = T^k(r, c), where T is one 90-degree CW rotation.
    #
    # So we can fill the final grid by:
    #   final[T^k(r, c)] = original[r, c]
    #
    # where k = (rI + 1) mod 4.  That yields an O(N^2) solution.

    # Prepare a list of lists for the final answer.
    final_grid = [list(["?"]*N) for _ in range(N)]

    def rotate90cw(r, c, N):
        """One 90-degree clockwise rotation in 0-based indexing."""
        return (c, N-1-r)

    for r in range(N):
        row_str = grid[r]
        for c in range(N):
            # Find how many times this cell will be rotated
            ring_index = min(r, c, N-1-r, N-1-c)
            k = (ring_index + 1) % 4  # how many 90-degree turns
            # Apply k rotations
            rr, cc = r, c
            for _ in range(k):
                rr, cc = rotate90cw(rr, cc, N)
            final_grid[rr][cc] = row_str[c]

    # Output the final grid
    out = []
    for r in range(N):
        out.append("".join(final_grid[r]))
    print("
".join(out))

# Do not forget to call main()
if __name__ == "__main__":
    main()