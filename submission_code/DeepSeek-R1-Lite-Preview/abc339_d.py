from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    N = int(data[0])
    grid = data[1:N+1]

    # Find positions of the two players
    pos1, pos2 = -1, -1
    for row in range(N):
        for col in range(N):
            if grid[row][col] == 'P':
                pos = row * N + col
                if pos1 == -1:
                    pos1 = pos
                else:
                    pos2 = pos
                    break
        if pos2 != -1:
            break

    # Ensure pos1 <= pos2
    if pos1 > pos2:
        pos1, pos2 = pos2, pos1

    # Initialize BFS
    queue = deque()
    queue.append((pos1, pos2, 0))
    visited_size = N * N * N * N
    visited = bytearray([0]) * visited_size  # Initialize with 0
    index = pos1 * N * N + pos2
    visited[index] = 1

    # Define directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        current_pos1, current_pos2, move_count = queue.popleft()

        if current_pos1 == current_pos2:
            print(move_count)
            return

        for dr, dc in directions:
            # Compute new position for player 1
            row1 = current_pos1 // N
            col1 = current_pos1 % N
            new_row1 = row1 + dr
            new_col1 = col1 + dc
            if 0 <= new_row1 < N and 0 <= new_col1 < N:
                if grid[new_row1][new_col1] == '.':
                    new_pos1 = new_row1 * N + new_col1
                else:
                    new_pos1 = current_pos1
            else:
                new_pos1 = current_pos1

            # Compute new position for player 2
            row2 = current_pos2 // N
            col2 = current_pos2 % N
            new_row2 = row2 + dr
            new_col2 = col2 + dc
            if 0 <= new_row2 < N and 0 <= new_col2 < N:
                if grid[new_row2][new_col2] == '.':
                    new_pos2 = new_row2 * N + new_col2
                else:
                    new_pos2 = current_pos2
            else:
                new_pos2 = current_pos2

            # Ensure pos1 <= pos2
            if new_pos1 > new_pos2:
                new_pos1, new_pos2 = new_pos2, new_pos1

            # Check if this state has been visited
            index = new_pos1 * N * N + new_pos2
            if visited[index] == 0:
                visited[index] = 1
                queue.append((new_pos1, new_pos2, move_count + 1))

    # If no meeting state is found
    print(-1)

if __name__ == "__main__":
    main()