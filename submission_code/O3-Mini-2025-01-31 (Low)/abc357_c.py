def main():
    import sys
    sys.setrecursionlimit(10000)
    N = int(sys.stdin.readline())
    
    size = 3 ** N
    
    # Check for each cell if it's white or black using a iterative process.
    def is_black(i, j):
        # Using property: if in any recursion step the cell is in middle block
        while i > 0 or j > 0:
            if i % 3 == 1 and j % 3 == 1:
                return False
            i //= 3
            j //= 3
        return True
    
    out_lines = []
    for i in range(size):
        row_chars = []
        for j in range(size):
            row_chars.append('#' if is_black(i, j) else '.')
        out_lines.append(''.join(row_chars))
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()