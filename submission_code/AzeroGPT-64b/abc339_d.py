from collections import deque

def find(player_start, player_end, obstacles):
    queue = deque([(player_start, 0)])
    visited = set()
    dx = [-1, 0, 1, 0]
    dy = [0, 1, 0, -1]

    while queue:
        pos, step = queue.popleft()
        if pos == player_end:
            return step
        visited.add(pos)
        for i in range(4):
            x = pos[0] + dx[i]
            y = pos[1] + dy[i]
            if 0 <= x < len(obstacles[0]) and 0 <= y < len(obstacles) and (x,y) not in obstacles and (x,y) not in visited:
                queue.append(((x,y), step + 1))
    return None


def solve_moving_players(N, paths):
    player_pos = None
    obstacles = set()
    
    for i in range(N):
        for j in range(N):
            if paths[i][j] == 'P':
                if not player_pos:
                    player_pos = [(i,j)]
                else:
                    player_pos.append((i,j))
            elif paths[i][j] == '#':
                obstacles.add((i,j))
    #player_pos = [(player, dest) for player, dest in zip(playerA, playerB)]

    distance = abs(player_pos[0][0]-player_pos[1][0]) + abs(player_pos[0][1]-player_pos[1][1])
    if distance > 1:
        result1 = find(player_pos[0], player_pos[1], obstacles)
        result2 = find(player_pos[1], player_pos[0], obstacles)
        if ((result1 % 2 == 1 and distance == 2) or (result2 % 2 == 1 and distance == 2)) and min(result1, result2) == None:
            return -1
        result = min(result1 if result1 is not None else 10**10, result2 if result2 is not None else 10**10) + distance
    else:
        result = distance
    
    return result

if __name__ == "__main__":
    N = int(input())
    paths = []
    for _ in range(N):
        paths.append([x for x in input().strip()])

    print(solve_moving_players(N, paths))