def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    n = 3 ** N  # grid size is 3^N

    # The idea: for a cell at (i, j) in the grid, if at any level of recursion
    # we find that both i and j mod 3 equal 1, then the cell is in the central white block.
    # If that never happens, the cell remains black.
    result = []
    for i in range(n):
        row_chars = []
        for j in range(n):
            x, y = i, j
            ch = '#'  # default black, as level-0 carpet
            # Check for each level if this cell falls into the white center.
            while x or y:
                if x % 3 == 1 and y % 3 == 1:
                    ch = '.'
                    break
                x //= 3
                y //= 3
            row_chars.append(ch)
        result.append("".join(row_chars))
    
    sys.stdout.write("
".join(result))

if __name__ == '__main__':
    main()