# YOUR CODE HERE
import collections

def solve():
    n = int(input())
    grid = [input() for _ in range(n)]
    
    players = []
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 'P':
                players.append((r, c))
    
    q = collections.deque([(players[0], players[1], 0)])
    visited = set()
    visited.add((players[0], players[1]))

    while q:
        p1, p2, moves = q.popleft()

        if p1 == p2:
            print(moves)
            return

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            np1 = list(p1)
            np2 = list(p2)

            if 0 <= np1[0] + dr < n and 0 <= np1[1] + dc < n and grid[np1[0] + dr][np1[1] + dc] != '#':
                np1[0] += dr
                np1[1] += dc
            
            if 0 <= np2[0] + dr < n and 0 <= np2[1] + dc < n and grid[np2[0] + dr][np2[1] + dc] != '#':
                np2[0] += dr
                np2[1] += dc
            
            np1 = tuple(np1)
            np2 = tuple(np2)

            if (np1, np2) not in visited:
                visited.add((np1, np2))
                q.append((np1, np2, moves + 1))
                
            if (np2, np1) not in visited:
                visited.add((np2, np1))
                q.append((np2, np1, moves + 1))
    
    print(-1)

solve()