def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().splitlines()
    n = int(input_data[0])
    grid = input_data[1:]
    
    players = []
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'P':
                players.append((i, j))
    if len(players) != 2:
        raise ValueError("There must be exactly two players.")
    start = (players[0], players[1])
    
    # 4 directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS over state space: state = (p1=(i1,j1), p2=(i2,j2))
    queue = deque()
    queue.append((start, 0))
    visited = set([start])
    
    def can_move(r, c):
        return 0 <= r < n and 0 <= c < n and grid[r][c] != '#'
    
    while queue:
        (p1, p2), moves = queue.popleft()
        # If players are on same cell, we are done:
        if p1 == p2:
            sys.stdout.write(str(moves))
            return
        
        for dr, dc in directions:
            # For each player, determine the new position if possible.
            new_p1 = p1
            nr, nc = p1[0] + dr, p1[1] + dc
            if can_move(nr, nc):
                new_p1 = (nr, nc)
            new_p2 = p2
            nr2, nc2 = p2[0] + dr, p2[1] + dc
            if can_move(nr2, nc2):
                new_p2 = (nr2, nc2)
            new_state = (new_p1, new_p2)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, moves + 1))
    
    sys.stdout.write("-1")

if __name__ == '__main__':
    main()