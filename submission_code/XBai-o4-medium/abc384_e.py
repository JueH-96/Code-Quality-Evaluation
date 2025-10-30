import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    H = int(input[idx]); idx += 1
    W = int(input[idx]); idx += 1
    X = int(input[idx]); idx += 1
    P = int(input[idx]) - 1; idx += 1  # convert to 0-based
    Q = int(input[idx]) - 1; idx += 1
    
    S = []
    for _ in range(H):
        row = list(map(int, input[idx:idx+W]))
        S.append(row)
        idx += W
    
    current_strength = S[P][Q]
    absorbed = [[False] * W for _ in range(H)]
    absorbed[P][Q] = True
    
    directions = [ (-1,0), (1,0), (0,-1), (0,1) ]
    heap = []
    
    # Initialize heap with adjacent cells of (P, Q)
    for dx, dy in directions:
        nx = P + dx
        ny = Q + dy
        if 0 <= nx < H and 0 <= ny < W:
            if not absorbed[nx][ny]:
                required = S[nx][ny] * X
                heapq.heappush(heap, (required, nx, ny))
    
    while heap:
        required, i, j = heapq.heappop(heap)
        if not absorbed[i][j] and current_strength > required:
            current_strength += S[i][j]
            absorbed[i][j] = True
            # Add adjacent cells to the heap
            for dx, dy in directions:
                nix = i + dx
                niy = j + dy
                if 0 <= nix < H and 0 <= niy < W:
                    if not absorbed[nix][niy]:
                        new_required = S[nix][niy] * X
                        heapq.heappush(heap, (new_required, nix, niy))
    
    print(current_strength)

if __name__ == "__main__":
    main()