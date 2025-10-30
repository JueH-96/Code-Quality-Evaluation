def main():
    import sys
    sys.setrecursionlimit(10000)
    
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    H = int(input_data[0])
    W = int(input_data[1])
    K = int(input_data[2])
    grid = input_data[3:]
    
    # Prepare board: grid[i][j] is char
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    answer = 0
    
    def dfs(i, j, steps, visited):
        nonlocal answer
        if steps == K:
            answer += 1
            return
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == '.' and (ni, nj) not in visited:
                    visited.add((ni, nj))
                    dfs(ni, nj, steps+1, visited)
                    visited.remove((ni, nj))
    
    # For each empty starting cell:
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                visited = {(i, j)}
                dfs(i, j, 0, visited)
                
    print(answer)
    
if __name__ == '__main__':
    main()