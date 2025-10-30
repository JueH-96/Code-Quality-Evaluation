def main():
    H, W, N = map(int, input().split())
    T = input()
    S = [input() for _ in range(H)]
    # Find the possible starting positions
    start_positions = []
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if S[i][j] == '.':
                start_positions.append((i, j))
    # Simulate the moves for each starting position
    possible_positions = set()
    for i, j in start_positions:
        x, y = i, j
        for move in T:
            if move == 'L':
                y -= 1
            elif move == 'R':
                y += 1
            elif move == 'U':
                x -= 1
            elif move == 'D':
                x += 1
            # Check if the next position is out of bounds or sea
            if x < 1 or x >= H - 1 or y < 1 or y >= W - 1 or S[x][y] == '#':
                break
        else:
            possible_positions.add((x, y))
    # Print the number of possible positions
    print(len(possible_positions))

main()