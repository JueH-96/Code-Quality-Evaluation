from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    m = int(data[1])
    grid = []
    index = 2
    for i in range(n):
        grid.append(data[index])
        index += 1
    
    touched = [[False for _ in range(m)] for _ in range(n)]
    processed = [[False for _ in range(m)] for _ in range(n)]
    
    # Starting position (2,2) in problem is (1,1) in 0-based index
    start_i, start_j = 1, 1
    if grid[start_i][start_j] == '.':
        touched[start_i][start_j] = True
        processed[start_i][start_j] = True
        queue = deque()
        queue.append((start_i, start_j))
    else:
        print(0)
        return
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        i, j = queue.popleft()
        
        for dx, dy in directions:
            path = []
            current_i, current_j = i, j
            path.append((current_i, current_j))
            
            while True:
                next_i = current_i + dx
                next_j = current_j + dy
                
                if next_i < 0 or next_i >= n or next_j < 0 or next_j >= m:
                    break
                if grid[next_i][next_j] == '#':
                    break
                
                current_i, current_j = next_i, next_j
                path.append((current_i, current_j))
            
            for x, y in path:
                if not touched[x][y]:
                    touched[x][y] = True
            
            end_i, end_j = current_i, current_j
            if not processed[end_i][end_j]:
                processed[end_i][end_j] = True
                queue.append((end_i, end_j))
    
    count = 0
    for i in range(n):
        for j in range(m):
            if grid[i][j] == '.' and touched[i][j]:
                count += 1
    print(count)

if __name__ == '__main__':
    main()