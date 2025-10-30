def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    N = int(sys.stdin.readline().strip())
    size = 3 ** N
    
    # Initialize grid with '.' (white cells)
    grid = [['.' for _ in range(size)] for _ in range(size)]
    
    def fill(r, c, n):
        """
        Recursively fill the grid to form a level-K carpet pattern.
        r, c: top-left coordinate of the current block
        n: the size of the current block (it will always be 3^k for some k)
        """
        if n == 1:
            # Base case: a single cell is entirely black
            grid[r][c] = '#'
            return
        
        step = n // 3
        for i in range(3):
            for j in range(3):
                nr = r + i * step
                nc = c + j * step
                # Skip the center block
                if i == 1 and j == 1:
                    continue
                fill(nr, nc, step)
    
    # For N=0, we just have a 1x1 black cell
    if N == 0:
        grid[0][0] = '#'
    else:
        fill(0, 0, size)
    
    # Print the grid
    for row in grid:
        print("".join(row))

# Do not forget to call main()
if __name__ == "__main__":
    main()