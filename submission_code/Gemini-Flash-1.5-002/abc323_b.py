# YOUR CODE HERE
n = int(input())
s = []
for _ in range(n):
    s.append(input())

wins = []
for i in range(n):
    win_count = 0
    for j in range(n):
        if i != j and s[i][j] == 'o':
            win_count += 1
    wins.append((win_count, i + 1))

wins.sort(reverse=True)

result = []
for win, player in wins:
    result.append(player)

print(*result)