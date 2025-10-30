from collections import deque

def main():
    H, W, Y = map(int, input().split())
    grid = [list(map(int, input().split())) for _ in range(H)]
    
    # Initialize the sea level
    sea_level = 0
    
    # To keep track of the remaining area
    remaining_area = H * W
    
    # To keep track of the sections that have sunk
    sunk = [[False for _ in range(W)] for _ in range(H)]
    
    # To process the sections in order of their elevation
    # We will use a list to store all sections with their elevation and coordinates
    sections = []
    for i in range(H):
        for j in range(W):
            sections.append((grid[i][j], i, j))
    
    # Sort the sections based on elevation
    sections.sort()
    
    # Initialize the index for the sections list
    idx = 0
    
    # Initialize the queue for BFS
    q = deque()
    
    # Iterate over each year
    for year in range(1, Y+1):
        sea_level = year
        # Process all sections with elevation <= sea_level that are not yet sunk
        while idx < len(sections) and sections[idx][0] <= sea_level:
            elevation, i, j = sections[idx]
            if not sunk[i][j]:
                # Check if it is adjacent to the sea or a sunk section
                if i == 0 or i == H-1 or j == 0 or j == W-1:
                    # It is on the border, so it will sink
                    q.append((i, j))
                    sunk[i][j] = True
                    remaining_area -= 1
                else:
                    # Check if any of its neighbors is sunk
                    if sunk[i-1][j] or sunk[i+1][j] or sunk[i][j-1] or sunk[i][j+1]:
                        q.append((i, j))
                        sunk[i][j] = True
                        remaining_area -= 1
            idx += 1
        # Now, perform BFS to sink all adjacent sections with elevation <= sea_level
        while q:
            i, j = q.popleft()
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if not sunk[ni][nj] and grid[ni][nj] <= sea_level:
                        sunk[ni][nj] = True
                        remaining_area -= 1
                        q.append((ni, nj))
        # Output the remaining area
        print(remaining_area)

if __name__ == "__main__":
    main()