from collections import deque

def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    grid = input_data[1:]
    
    # Find the two players' initial positions
    players = []
    for r in range(N):
        for c in range(N):
            if grid[r][c] == 'P':
                players.append((r, c))
    (r1, c1), (r2, c2) = players[0], players[1]
    
    # Directions for movement: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Check if a position is in bounds and not an obstacle
    def can_move(r, c):
        if 0 <= r < N and 0 <= c < N and grid[r][c] != '#':
            return True
        return False
    
    # BFS in state space (r1, c1, r2, c2)
    visited = [[[[False]*N for _ in range(N)] for _ in range(N)] for _ in range(N)]
    visited[r1][c1][r2][c2] = True
    
    queue = deque()
    queue.append((r1, c1, r2, c2, 0))
    
    while queue:
        rr1, cc1, rr2, cc2, dist = queue.popleft()
        
        # If both players are in the same cell, we are done
        if rr1 == rr2 and cc1 == cc2:
            print(dist)
            return
        
        for dr, dc in directions:
            # Move player 1 if possible
            nr1, nc1 = rr1, cc1
            if can_move(rr1 + dr, cc1 + dc):
                nr1, nc1 = rr1 + dr, cc1 + dc
            
            # Move player 2 if possible
            nr2, nc2 = rr2, cc2
            if can_move(rr2 + dr, cc2 + dc):
                nr2, nc2 = rr2 + dr, cc2 + dc
            
            # Check if we have visited this new state
            if not visited[nr1][nc1][nr2][nc2]:
                visited[nr1][nc1][nr2][nc2] = True
                queue.append((nr1, nc1, nr2, nc2, dist+1))
    
    # If we exhaust all states and cannot bring them together
    print(-1)

def main():
    solve()

if __name__ == "__main__":
    main()