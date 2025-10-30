import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    people = []
    sum_b = 0
    teams = [0, 0, 0]
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        a -= 1  # 0-based for easier handling
        people.append((a, b))
        sum_b += b
        teams[a] += 1  # track initial team counts
    
    if sum_b % 3 != 0:
        print(-1)
        return
    
    target = sum_b // 3
    s = [0, 0, 0]
    for a, b in people:
        s[a] += b
    d1 = target - s[0]
    d2 = target - s[1]
    mask = 0
    if teams[0] > 0:
        mask |= 1 << 0
    if teams[1] > 0:
        mask |= 1 << 1
    if teams[2] > 0:
        mask |= 1 << 2
    initial_mask = mask
    
    if d1 == 0 and d2 == 0 and mask == 0b111:
        print(0)
        return
    
    visited = {}
    queue = deque()
    queue.append((d1, d2, mask, 0))
    visited[(d1, d2, mask)] = 0
    
    while queue:
        current_d1, current_d2, current_mask, moves = queue.popleft()
        
        for i in range(N):
            a, b = people[i]
            for new_a in range(3):
                if new_a == a:
                    continue
                
                X = a
                Y = new_a
                
                delta_X = b
                delta_Y = -b
                
                new_d1 = current_d1
                new_d2 = current_d2
                
                if X == 0:
                    new_d1 += delta_X
                elif X == 1:
                    new_d2 += delta_X
                else:
                    pass
                
                if Y == 0:
                    new_d1 -= delta_Y
                elif Y == 1:
                    new_d2 -= delta_Y
                else:
                    pass
                
                new_teams = [0, 0, 0]
                new_teams[0] = teams[0] - (1 if X == 0 else 0) + (1 if Y == 0 else 0)
                new_teams[1] = teams[1] - (1 if X == 1 else 0) + (1 if Y == 1 else 0)
                new_teams[2] = teams[2] - (1 if X == 2 else 0) + (1 if Y == 2 else 0)
                
                new_mask = 0
                if new_teams[0] > 0:
                    new_mask |= 1 << 0
                if new_teams[1] > 0:
                    new_mask |= 1 << 1
                if new_teams[2] > 0:
                    new_mask |= 1 << 2
                
                if new_d1 == 0 and new_d2 == 0 and new_mask == 0b111:
                    print(moves + 1)
                    return
                
                key = (new_d1, new_d2, new_mask)
                if key not in visited or visited[key] > moves + 1:
                    visited[key] = moves + 1
                    queue.append((new_d1, new_d2, new_mask, moves + 1))
    
    print(-1)

if __name__ == "__main__":
    main()