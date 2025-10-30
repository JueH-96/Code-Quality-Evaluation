# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    S_x = int(data[2])
    S_y = int(data[3])
    
    houses = set()
    index = 4
    for _ in range(N):
        X = int(data[index])
        Y = int(data[index + 1])
        houses.add((X, Y))
        index += 2
    
    moves = []
    for _ in range(M):
        D = data[index]
        C = int(data[index + 1])
        moves.append((D, C))
        index += 2
    
    current_x, current_y = S_x, S_y
    visited_houses = set()
    
    for D, C in moves:
        if D == 'U':
            for y in range(current_y + 1, current_y + C + 1):
                if (current_x, y) in houses:
                    visited_houses.add((current_x, y))
            current_y += C
        elif D == 'D':
            for y in range(current_y - 1, current_y - C - 1, -1):
                if (current_x, y) in houses:
                    visited_houses.add((current_x, y))
            current_y -= C
        elif D == 'L':
            for x in range(current_x - 1, current_x - C - 1, -1):
                if (x, current_y) in houses:
                    visited_houses.add((x, current_y))
            current_x -= C
        elif D == 'R':
            for x in range(current_x + 1, current_x + C + 1):
                if (x, current_y) in houses:
                    visited_houses.add((x, current_y))
            current_x += C
    
    print(current_x, current_y, len(visited_houses))

if __name__ == "__main__":
    main()