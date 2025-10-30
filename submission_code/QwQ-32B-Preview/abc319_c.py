def main():
    import sys
    from fractions import Fraction

    # Read the grid
    grid = []
    for _ in range(3):
        row = list(map(int, sys.stdin.readline().split()))
        grid.append(row)

    # Define all lines: rows, columns, diagonals
    lines = [
        # Rows
        [(0,0),(0,1),(0,2)],
        [(1,0),(1,1),(1,2)],
        [(2,0),(2,1),(2,2)],
        # Columns
        [(0,0),(1,0),(2,0)],
        [(0,1),(1,1),(2,1)],
        [(0,2),(1,2),(2,2)],
        # Diagonals
        [(0,0),(1,1),(2,2)],
        [(0,2),(1,1),(2,0)]
    ]

    # Function to check if a line has exactly two same numbers
    def is_disappointing_line(line):
        a = grid[line[0][0]][line[0][1]]
        b = grid[line[1][0]][line[1][1]]
        c = grid[line[2][0]][line[2][1]]
        numbers = [a, b, c]
        freq = {}
        for num in numbers:
            freq[num] = freq.get(num, 0) + 1
        return sorted(freq.values()) == [1, 2]

    # Count the number of disappointing lines
    n = 0
    for line in lines:
        if is_disappointing_line(line):
            n += 1

    # Calculate probability
    prob = (Fraction(2, 3)) ** n
    # Convert to float for printing
    prob_float = float(prob)
    
    # Print the probability with high precision
    print("{0:.30f}".format(prob_float))

if __name__ == "__main__":
    main()