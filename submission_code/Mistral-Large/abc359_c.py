import sys

def min_toll(S_x, S_y, T_x, T_y):
    # Calculate the initial and final tiles
    start_tile_x = S_x // 2
    start_tile_y = S_y // 2
    end_tile_x = T_x // 2
    end_tile_y = T_y // 2

    # Calculate the minimum number of vertical and horizontal moves
    vertical_moves = abs(end_tile_y - start_tile_y)
    horizontal_moves = abs(end_tile_x - start_tile_x)

    # The minimum toll is the sum of vertical and horizontal moves
    min_toll = vertical_moves + horizontal_moves

    return min_toll

def main():
    input = sys.stdin.read
    data = input().split()

    S_x = int(data[0])
    S_y = int(data[1])
    T_x = int(data[2])
    T_y = int(data[3])

    result = min_toll(S_x, S_y, T_x, T_y)
    print(result)

if __name__ == "__main__":
    main()