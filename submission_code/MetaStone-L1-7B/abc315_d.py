from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = data[2:2+H]
    
    row_colors = []
    for row in grid:
        row_colors.append(set(row))
    
    column_colors = [set() for _ in range(W)]
    for j in range(W):
        for i in range(H):
            column_colors[j].add(grid[i][j])
    
    removed = [[False for _ in range(W)] for _ in range(H)]
    queue = deque()
    
    for i in range(H):
        if len(row_colors[i]) == 1:
            queue.append(i)
    
    for j in range(W):
        if len(column_colors[j]) == 1:
            queue.append(j)
    
    while queue:
        # Process rows
        for row in queue:
            if removed[row]:
                continue
            for j in range(W):
                if not removed[row][j]:
                    removed[row][j] = True
            for j in range(W):
                if not removed[row][j]:
                    c = grid[row][j]
                    column_colors[j].add(c)
                    if len(column_colors[j]) == 1:
                        if j not in queue:
                            queue.append(j)
        
        # Process columns
        while queue:
            col = queue.popleft()
            if removed[col]:
                continue
            for i in range(H):
                if not removed[i][col]:
                    removed[i][col] = True
            for i in range(H):
                if not removed[i][col]:
                    c = grid[i][col]
                    row_colors[i].add(c)
                    if len(row_colors[i]) == 1:
                        if i not in queue:
                            queue.append(i)
    
    count = 0
    for i in range(H):
        for j in range(W):
            if not removed[i][j]:
                count += 1
    print(count)

if __name__ == "__main__":
    main()