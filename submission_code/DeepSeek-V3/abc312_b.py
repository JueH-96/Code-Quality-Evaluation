# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    grid = data[2:2+N]
    
    # Define the regions that must be black and white
    # For the top-left 3x3 and bottom-right 3x3 regions
    # For the cells adjacent to these regions
    
    # Precompute the positions for the 9x9 region
    # For each possible top-left (i,j) of the 9x9 region
    # Check if the region is entirely within the grid
    # Then check the conditions
    
    # Precompute the positions for the black and white cells
    # For the top-left 3x3 region: rows 0-2, cols 0-2
    # For the bottom-right 3x3 region: rows 6-8, cols 6-8
    # For the adjacent cells: 
    # For top-left 3x3: rows 0-3, cols 0-3, excluding the 3x3 itself
    # For bottom-right 3x3: rows 5-8, cols 5-8, excluding the 3x3 itself
    
    # So, for each (i,j) as top-left of 9x9 region:
    # Check if i+8 < N and j+8 < M
    # Then, for the top-left 3x3: grid[i+x][j+y] == '#' for x in 0-2, y in 0-2
    # For the bottom-right 3x3: grid[i+6+x][j+6+y] == '#' for x in 0-2, y in 0-2
    # For the adjacent cells:
    # For top-left 3x3: cells in rows i-1 to i+3, cols j-1 to j+3, excluding the 3x3 itself
    # For bottom-right 3x3: cells in rows i+5 to i+9, cols j+5 to j+9, excluding the 3x3 itself
    # But since the region is 9x9, the adjacent cells are within the 9x9 region
    
    # So, for the top-left 3x3, the adjacent cells are:
    # rows i to i+3, cols j to j+3, excluding the 3x3 itself
    # Similarly for the bottom-right 3x3
    
    # So, for each (i,j):
    # Check if i+8 < N and j+8 < M
    # Then, for the top-left 3x3:
    # grid[i+x][j+y] == '#' for x in 0-2, y in 0-2
    # For the bottom-right 3x3:
    # grid[i+6+x][j+6+y] == '#' for x in 0-2, y in 0-2
    # For the adjacent cells:
    # For top-left 3x3:
    # cells in rows i to i+3, cols j to j+3, excluding the 3x3 itself
    # For bottom-right 3x3:
    # cells in rows i+5 to i+9, cols j+5 to j+9, excluding the 3x3 itself
    
    # So, for the top-left 3x3, the adjacent cells are:
    # rows i to i+3, cols j to j+3, excluding the 3x3 itself
    # So, for x in 0-3, y in 0-3, but not (x in 0-2 and y in 0-2)
    # Similarly for the bottom-right 3x3
    
    # So, for each (i,j):
    # Check if i+8 < N and j+8 < M
    # Then, for the top-left 3x3:
    # grid[i+x][j+y] == '#' for x in 0-2, y in 0-2
    # For the bottom-right 3x3:
    # grid[i+6+x][j+6+y] == '#' for x in 0-2, y in 0-2
    # For the adjacent cells:
    # For top-left 3x3:
    # for x in 0-3:
    #   for y in 0-3:
    #       if not (x in 0-2 and y in 0-2):
    #           if grid[i+x][j+y] != '.':
    #               break
    # Similarly for the bottom-right 3x3
    
    # So, implement this logic
    
    results = []
    
    for i in range(N - 8):
        for j in range(M - 8):
            # Check top-left 3x3
            top_left_black = True
            for x in range(3):
                for y in range(3):
                    if grid[i+x][j+y] != '#':
                        top_left_black = False
                        break
                if not top_left_black:
                    break
            if not top_left_black:
                continue
            # Check bottom-right 3x3
            bottom_right_black = True
            for x in range(3):
                for y in range(3):
                    if grid[i+6+x][j+6+y] != '#':
                        bottom_right_black = False
                        break
                if not bottom_right_black:
                    break
            if not bottom_right_black:
                continue
            # Check adjacent cells for top-left 3x3
            adjacent_white_top_left = True
            for x in range(4):
                for y in range(4):
                    if not (x < 3 and y < 3):
                        if grid[i+x][j+y] != '.':
                            adjacent_white_top_left = False
                            break
                if not adjacent_white_top_left:
                    break
            if not adjacent_white_top_left:
                continue
            # Check adjacent cells for bottom-right 3x3
            adjacent_white_bottom_right = True
            for x in range(4):
                for y in range(4):
                    if not (x < 3 and y < 3):
                        if grid[i+5+x][j+5+y] != '.':
                            adjacent_white_bottom_right = False
                            break
                if not adjacent_white_bottom_right:
                    break
            if not adjacent_white_bottom_right:
                continue
            # If all conditions are satisfied, add to results
            results.append((i+1, j+1))
    
    # Sort the results
    results.sort()
    
    # Print the results
    for res in results:
        print(res[0], res[1])

if __name__ == "__main__":
    main()