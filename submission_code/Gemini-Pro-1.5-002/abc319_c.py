# YOUR CODE HERE
grid = []
for _ in range(3):
    grid.append(list(map(int, input().split())))

lines = []
for i in range(3):
    lines.append(grid[i])
for j in range(3):
    lines.append([grid[i][j] for i in range(3)])
lines.append([grid[i][i] for i in range(3)])
lines.append([grid[i][2 - i] for i in range(3)])

count = 0
for line in lines:
    if len(set(line)) == 2:
        if line[0] == line[1]:
            count += 1
        elif line[1] == line[2]:
            count +=1
        elif line[0] == line[2]:
            count += 1
    elif len(set(line)) == 1:
        count += 3
        
print(1 - count / 6)