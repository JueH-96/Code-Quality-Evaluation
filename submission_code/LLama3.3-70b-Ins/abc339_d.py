from collections import deque

def solve():
    n = int(input())
    grid = [list(input()) for _ in range(n)]
    players = []
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'P':
                players.append((i, j))

    queue = deque([(players, 0)])
    visited = {tuple(map(tuple, players))}

    while queue:
        players, moves = queue.popleft()
        if players[0] == players[1]:
            return moves

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_players = [list(player) for player in players]
            for i, (x, y) in enumerate(players):
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] != '#':
                    new_players[i] = [nx, ny]
            new_players_tuple = tuple(map(tuple, new_players))
            if new_players_tuple not in visited:
                queue.append((new_players, moves + 1))
                visited.add(new_players_tuple)

    return -1

print(solve())