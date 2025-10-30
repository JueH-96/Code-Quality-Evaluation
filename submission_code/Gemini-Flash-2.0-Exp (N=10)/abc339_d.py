from collections import deque

def solve():
    n = int(input())
    grid = [input() for _ in range(n)]
    
    players = []
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 'P':
                players.append((r, c))
    
    start_p1 = players[0]
    start_p2 = players[1]
    
    q = deque([(start_p1, start_p2, 0)])
    visited = set()
    visited.add((start_p1, start_p2))
    
    while q:
        p1, p2, moves = q.popleft()
        
        if p1 == p2:
            print(moves)
            return
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dr, dc in directions:
            new_p1_r = p1[0] + dr
            new_p1_c = p1[1] + dc
            
            new_p2_r = p2[0] + dr
            new_p2_c = p2[1] + dc
            
            
            if 0 <= new_p1_r < n and 0 <= new_p1_c < n and grid[new_p1_r][new_p1_c] != '#':
                next_p1 = (new_p1_r, new_p1_c)
            else:
                next_p1 = p1
                
            if 0 <= new_p2_r < n and 0 <= new_p2_c < n and grid[new_p2_r][new_p2_c] != '#':
                next_p2 = (new_p2_r, new_p2_c)
            else:
                next_p2 = p2
            
            if (next_p1, next_p2) not in visited:
                visited.add((next_p1, next_p2))
                q.append((next_p1, next_p2, moves + 1))
    
    print("-1")

solve()