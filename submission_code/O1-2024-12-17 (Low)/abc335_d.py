def main():
    import sys
    sys.setrecursionlimit(10**7)

    N = int(sys.stdin.readline().strip())
    # We know N is odd and 3 <= N <= 45.

    # Center (using 0-based indexing):
    rC = N // 2
    cC = N // 2

    #--------------------------------------------------------------------
    # 1) First, build a "snake path" covering all N*N cells exactly once.
    #    We'll do the usual row-by-row "zigzag":
    #
    #    Row 0: left -> right  ( (0,0) -> (0,1) -> ... -> (0,N-1) )
    #    Row 1: right -> left  ( (1,N-1) -> (1,N-2) -> ... -> (1,0) )
    #    Row 2: left -> right
    #    etc.
    #
    #    This yields a path of length N^2, with each consecutive pair
    #    of cells edge-adjacent.
    #--------------------------------------------------------------------

    path = []
    for r in range(N):
        if r % 2 == 0:
            # left-to-right
            for c in range(N):
                path.append((r, c))
        else:
            # right-to-left
            for c in range(N - 1, -1, -1):
                path.append((r, c))

    #--------------------------------------------------------------------
    # 2) Find where the center cell (rC,cC) appears in that path.
    #    We want ultimately to "remove" the center from this path
    #    so it can be Takahashi's location, and keep the rest
    #    as a Hamiltonian path of length N^2-1.
    #--------------------------------------------------------------------
    idx_center = path.index((rC, cC))  # guaranteed to be in the path

    #--------------------------------------------------------------------
    # 3) We rotate the path so that (rC,cC) becomes the LAST cell in the path.
    #    Then we pop it off, leaving a path of length N^2 - 1 that visits
    #    every other cell exactly once in a valid edge-adjacent order.
    #    
    #    Rotation trick: 
    #        new_path = path[idx_center+1:] + path[:idx_center+1]
    #    But we want the center to be last, so we do:
    #        new_path = path[idx_center+1:] + path[:idx_center]
    #        then (rC,cC) is the last of the old slice[:idx_center+1].
    #    We'll do it carefully below.
    #--------------------------------------------------------------------
    # One straightforward way: put center at the end, so the path up to
    # that point is everything else.
    new_path = path[idx_center+1:] + path[:idx_center] 
    # now path[idx_center] == (rC,cC) is effectively removed from the middle
    # and we have not yet appended it.  Let's append now so it's last:
    new_path.append( (rC,cC) )
    # Now the last cell is center.

    # We then pop the last cell (center) out to "remove" it from the path:
    new_path.pop()  # remove center so that final path has length N^2-1

    #--------------------------------------------------------------------
    # 4) Assign numbers 1..(N^2 - 1) to this path in order.
    #    Then put T in the center cell.
    #--------------------------------------------------------------------
    # Prepare an output grid:
    grid = [[0]*N for _ in range(N)]

    # Place T in center:
    grid[rC][cC] = "T"

    # Place the parts 1..N^2-1 along new_path:
    for i, (r, c) in enumerate(new_path):
        grid[r][c] = i+1

    #--------------------------------------------------------------------
    # 5) Print the result
    #--------------------------------------------------------------------
    for r in range(N):
        print(" ".join(str(grid[r][c]) for c in range(N)))


# Don't forget to call main():
if __name__ == "__main__":
    main()