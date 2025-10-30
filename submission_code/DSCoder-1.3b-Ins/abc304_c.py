import sys
import math

def read_input():
    N, D = map(int, sys.stdin.readline().split())
    points = []
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().split())
        points.append((x, y))
    return N, D, points

def solve(N, D, points):
    for i in range(N):
        infected = True
        for j in range(N):
            if i != j:
                distance = math.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                if distance <= D:
                    infected = False
                    break
        print('Yes' if infected else 'No')

def main():
    N, D, points = read_input()
    solve(N, D, points)

if __name__ == "__main__":
    main()