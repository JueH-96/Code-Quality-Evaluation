def count_safe_squares(N, M, pieces):
    # Set to store all the squares that can be attacked
    attacked_squares = set()

    # Directions a knight can attack from its position
    knight_moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]

    # For each piece, calculate the attacked squares
    for a, b in pieces:
        attacked_squares.add((a, b))  # The square itself is attacked
        for dx, dy in knight_moves:
            x, y = a + dx, b + dy
            if 1 <= x <= N and 1 <= y <= N:
                attacked_squares.add((x, y))

    # Total squares in the grid
    total_squares = N * N

    # The number of safe squares is total squares minus attacked squares
    safe_squares = total_squares - len(attacked_squares)

    return safe_squares

import sys

def main():
    input_data = sys.stdin.read().strip().splitlines()
    first_line = input_data[0].split()
    N = int(first_line[0])
    M = int(first_line[1])
    
    pieces = []
    for i in range(1, M + 1):
        a, b = map(int, input_data[i].split())
        pieces.append((a, b))

    result = count_safe_squares(N, M, pieces)
    print(result)

if __name__ == "__main__":
    main()