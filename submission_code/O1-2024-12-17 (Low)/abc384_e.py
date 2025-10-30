import sys
import heapq

def main():
    input_data = sys.stdin.read().strip().split()
    H = int(input_data[0])
    W = int(input_data[1])
    X = int(input_data[2])
    P = int(input_data[3]) - 1  # convert to 0-based
    Q = int(input_data[4]) - 1  # convert to 0-based
    
    S = []
    idx = 5
    for _ in range(H):
        row = list(map(int, input_data[idx:idx+W]))
        S.append(row)
        idx += W

    # Directions for adjacency (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # visited[i][j] = True means the slime in cell (i, j) is absorbed
    visited = [[False]*W for _ in range(H)]
    visited[P][Q] = True
    
    # Priority queue (min-heap) for adjacent slimes: (slime_strength, i, j)
    pq = []
    
    # Add all neighbors of (P, Q) to the priority queue
    for di, dj in directions:
        ni, nj = P + di, Q + dj
        if 0 <= ni < H and 0 <= nj < W:
            heapq.heappush(pq, (S[ni][nj], ni, nj))
    
    takahashi_strength = S[P][Q]
    
    in_queue = [[False]*W for _ in range(H)]
    for _, i, j in pq:
        in_queue[i][j] = True
    
    # Absorb slimes while possible
    while pq:
        s_val, i, j = heapq.heappop(pq)
        in_queue[i][j] = False
        # Check if we can absorb this slime
        # Condition: s_val * X < takahashi_strength  (instead of s_val < takahashi_strength / X)
        if s_val * X < takahashi_strength:
            # Absorb
            visited[i][j] = True
            takahashi_strength += s_val
            # Add newly adjacent slimes
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if not visited[ni][nj] and not in_queue[ni][nj]:
                        heapq.heappush(pq, (S[ni][nj], ni, nj))
                        in_queue[ni][nj] = True
        else:
            # We cannot absorb this slime, and since it's the smallest in the heap,
            # we cannot absorb any others. Stop.
            break
    
    print(takahashi_strength)

# Call main() to solve the problem
if __name__ == "__main__":
    main()