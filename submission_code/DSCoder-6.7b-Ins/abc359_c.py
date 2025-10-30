def min_toll(S_x, S_y, T_x, T_y):
    # Calculate the difference in x and y coordinates
    dx = T_x - S_x
    dy = T_y - S_y

    # Calculate the minimum number of moves in x and y directions
    moves_x = (dx + 1) // 2
    moves_y = (dy + 1) // 2

    # Calculate the total number of moves
    total_moves = max(moves_x, moves_y)

    # Calculate the total toll
    total_toll = total_moves

    return total_toll

# Read the inputs from stdin
S_x, S_y, T_x, T_y = map(int, input().split())

# Solve the problem and write the answer to stdout
print(min_toll(S_x, S_y, T_x, T_y))