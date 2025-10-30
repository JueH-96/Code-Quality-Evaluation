def solve():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    grid = input_data[2:]
    
    # Helper to check if a cell is black
    def is_black(r, c):
        return grid[r][c] == '#'
    
    # Helper to check if a cell is white
    def is_white(r, c):
        return grid[r][c] == '.'
    
    ans = []
    
    # We only need to consider top-left corners of 9x9 subgrids
    for i in range(N - 8):
        for j in range(M - 8):
            # Check top-left 3x3 region is all black
            valid = True
            for r in range(i, i+3):
                for c in range(j, j+3):
                    if not is_black(r, c):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Check bottom-right 3x3 region is all black
            for r in range(i+6, i+9):
                for c in range(j+6, j+9):
                    if not is_black(r, c):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Check the 7 adjacency cells around top-left 3x3 are white
            #  - cells in subgrid coords [0..2,3], [3,0..2], [3,3]
            #  which in absolute coords becomes:
            #  (r, j+3) for r in [i..i+2],
            #  (i+3, c) for c in [j..j+2],
            #  (i+3, j+3)
            for r in range(i, i+3):
                if not is_white(r, j+3):
                    valid = False
                    break
            if valid:
                for c in range(j, j+3):
                    if not is_white(i+3, c):
                        valid = False
                        break
            if valid:
                if not is_white(i+3, j+3):
                    valid = False
            if not valid:
                continue
            
            # Check the 7 adjacency cells around bottom-right 3x3 are white
            #  - subgrid coords [6..8,5], [5,6..8], [5,5]
            #  in absolute coords:
            #  (r, j+5) for r in [i+6..i+8],
            #  (i+5, c) for c in [j+6..j+8],
            #  (i+5, j+5)
            for r in range(i+6, i+9):
                if not is_white(r, j+5):
                    valid = False
                    break
            if valid:
                for c in range(j+6, j+9):
                    if not is_white(i+5, c):
                        valid = False
                        break
            if valid:
                if not is_white(i+5, j+5):
                    valid = False
            if not valid:
                continue
            
            # If all checks passed, record this top-left corner (1-based index)
            ans.append((i+1, j+1))
    
    # Output the results
    for r, c in ans:
        print(r, c)

def main():
    solve()

# For the testing environment where only solve() is called:
if __name__ == "__main__":
    main()