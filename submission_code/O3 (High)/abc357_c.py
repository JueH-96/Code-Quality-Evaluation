import sys

def main() -> None:
    # Read level N
    N_line = sys.stdin.readline()
    if not N_line:
        return
    N = int(N_line.strip())
    
    side = 3 ** N          # side length of the carpet
    
    # Generate and immediately print each row
    for i in range(side):
        row_chars = []
        for j in range(side):
            x, y = i, j
            white = False
            # Check every level from finest to coarsest
            while x or y:
                if x % 3 == 1 and y % 3 == 1:   # central cell at this level
                    white = True
                    break
                x //= 3
                y //= 3
            row_chars.append('.' if white else '#')
        # Output the constructed row
        sys.stdout.write(''.join(row_chars))
        if i != side - 1:
            sys.stdout.write('
')

if __name__ == "__main__":
    main()