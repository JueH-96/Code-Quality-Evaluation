def main():
    import sys
    
    input_data = sys.stdin.read().strip()
    N = int(input_data)
    size = 3 ** N
    
    def cell_color(r, c):
        # Determine if cell at (r,c) is black (#) or white (.)
        # by checking if it ever falls into a center block at any recursion level.
        while r > 0 or c > 0:
            if r % 3 == 1 and c % 3 == 1:
                return '.'
            r //= 3
            c //= 3
        return '#'
    
    for i in range(size):
        row = [cell_color(i, j) for j in range(size)]
        print("".join(row))

# Do not remove this call to main
main()