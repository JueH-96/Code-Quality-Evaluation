def main():
    import sys
    import math

    # Read input
    S_x, S_y, T_x, T_y = map(int, sys.stdin.read().split())

    # Function to calculate tile coordinates
    def get_tile_coordinates(x, y):
        i = math.floor(x)
        j = math.floor(y)
        col = (i - j) // 2
        row = (i + j) // 2
        return (col, row)

    # Get tile coordinates for start and end
    start_col, start_row = get_tile_coordinates(S_x, S_y)
    end_col, end_row = get_tile_coordinates(T_x, T_y)

    # Calculate the differences
    delta_col = end_col - start_col
    delta_row = end_row - start_row

    # Calculate minimal toll (Manhattan distance)
    minimal_toll = abs(delta_col) + abs(delta_row)

    # Output the result
    print(minimal_toll)

if __name__ == "__main__":
    main()