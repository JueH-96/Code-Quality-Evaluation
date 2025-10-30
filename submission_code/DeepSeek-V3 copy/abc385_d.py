import sys
import math

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    S_x = int(data[idx+2])
    S_y = int(data[idx+3])
    idx += 4
    houses = []
    for _ in range(N):
        X = int(data[idx])
        Y = int(data[idx+1])
        houses.append((X, Y))
        idx += 2
    moves = []
    for _ in range(M):
        D = data[idx]
        C = int(data[idx+1])
        moves.append((D, C))
        idx += 2
    
    current_x, current_y = S_x, S_y
    visited = set()
    
    for D, C in moves:
        if D == 'U':
            new_x = current_x
            new_y = current_y + C
        elif D == 'D':
            new_x = current_x
            new_y = current_y - C
        elif D == 'L':
            new_x = current_x - C
            new_y = current_y
        elif D == 'R':
            new_x = current_x + C
            new_y = current_y
        
        # Check if any house lies on the line segment from (current_x, current_y) to (new_x, new_y)
        # The line segment can be parameterized as (x(t), y(t)) = (current_x + t*(new_x - current_x), current_y + t*(new_y - current_y)), t in [0,1]
        # For each house (X, Y), check if there exists t in [0,1] such that X = current_x + t*(new_x - current_x) and Y = current_y + t*(new_y - current_y)
        # Solving for t: t = (X - current_x) / (new_x - current_x) if new_x != current_x, else t = (Y - current_y) / (new_y - current_y)
        # Then check if t is in [0,1] and if the other coordinate matches
        
        for X, Y in houses:
            if new_x == current_x:
                if X == current_x:
                    t = (Y - current_y) / (new_y - current_y)
                    if 0 <= t <= 1:
                        visited.add((X, Y))
            elif new_y == current_y:
                if Y == current_y:
                    t = (X - current_x) / (new_x - current_x)
                    if 0 <= t <= 1:
                        visited.add((X, Y))
            else:
                t_x = (X - current_x) / (new_x - current_x)
                t_y = (Y - current_y) / (new_y - current_y)
                if abs(t_x - t_y) < 1e-9 and 0 <= t_x <= 1:
                    visited.add((X, Y))
        
        current_x, current_y = new_x, new_y
    
    print(current_x, current_y, len(visited))

if __name__ == "__main__":
    main()