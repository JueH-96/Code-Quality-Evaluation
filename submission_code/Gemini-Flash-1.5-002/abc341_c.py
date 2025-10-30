H, W, N = map(int, input().split())
T = input()
S = [input() for _ in range(H)]

possible_positions = []
for start_row in range(H):
    for start_col in range(W):
        if S[start_row][start_col] == '.':
            row, col = start_row, start_col
            valid = True
            path = [(row, col)]
            for move in T:
                if move == 'L':
                    col -= 1
                elif move == 'R':
                    col += 1
                elif move == 'U':
                    row -= 1
                elif move == 'D':
                    row += 1
                if not (0 <= row < H and 0 <= col < W and S[row][col] == '.'):
                    valid = False
                    break
                path.append((row, col))
            if valid:
                possible_positions.append((row, col))

count = {}
for pos in possible_positions:
    if pos not in count:
        count[pos] = 0
    count[pos] += 1

print(len(count))