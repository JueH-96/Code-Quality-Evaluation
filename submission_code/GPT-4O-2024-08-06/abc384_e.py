def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    X = int(data[2])
    P = int(data[3]) - 1  # Convert to zero-based index
    Q = int(data[4]) - 1  # Convert to zero-based index
    
    S = []
    index = 5
    for _ in range(H):
        S.append([int(data[index + j]) for j in range(W)])
        index += W
    
    from collections import deque
    
    # Directions for moving in the grid: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Initialize the BFS
    queue = deque([(P, Q)])
    visited = [[False] * W for _ in range(H)]
    visited[P][Q] = True
    takahashi_strength = S[P][Q]
    
    while queue:
        i, j = queue.popleft()
        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj]:
                if S[ni][nj] < takahashi_strength / X:
                    takahashi_strength += S[ni][nj]
                    visited[ni][nj] = True
                    queue.append((ni, nj))
    
    print(takahashi_strength)