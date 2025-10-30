def solve():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    S = input_data[2:]

    # We'll use 0-based indexing internally.
    # The player starts at (1,1) in 0-based, which corresponds to (2,2) in 1-based.
    start_r, start_c = 1, 1

    # Directions: right, left, down, up
    directions = [(0,1), (0,-1), (1,0), (-1,0)]
    
    visited = [[False]*M for _ in range(N)]
    visited[start_r][start_c] = True

    queue = deque()
    queue.append((start_r, start_c))

    count = 1  # We start on that ice square, so we've touched 1 square.

    while queue:
        r, c = queue.popleft()
        # Try all four directions
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            # Move in that direction step by step until hitting a rock
            while True:
                if S[nr][nc] == '#':
                    # We hit a rock, so we stop
                    break
                # It's ice
                if not visited[nr][nc]:
                    visited[nr][nc] = True
                    count += 1
                    queue.append((nr, nc))
                # Step further in the same direction
                nr += dr
                nc += dc

    print(count)

def main():
    solve()

if __name__ == "__main__":
    main()