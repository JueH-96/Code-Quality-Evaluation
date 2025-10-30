# YOUR CODE HERE
N = int(input())
results = []
for _ in range(N):
    results.append(input().strip())

# Count wins for each player
wins = []
for i in range(N):
    win_count = 0
    for j in range(N):
        if results[i][j] == 'o':
            win_count += 1
    wins.append((win_count, i + 1))  # (wins, player_number)

# Sort by wins (descending) and then by player number (ascending)
wins.sort(key=lambda x: (-x[0], x[1]))

# Extract player numbers
answer = [str(x[1]) for x in wins]

# Print the result
print(' '.join(answer))