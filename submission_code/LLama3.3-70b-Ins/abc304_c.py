import sys
import math

def calculate_distance(x1, y1, x2, y2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def spread_virus(N, D, coordinates):
    """Spread the virus to people within a distance of D from an infected person."""
    infected = [False] * N
    infected[0] = True  # Person 1 is initially infected

    # Continue spreading the virus until no new people are infected
    while True:
        new_infections = 0
        for i in range(N):
            if infected[i]:
                for j in range(N):
                    if not infected[j]:
                        distance = calculate_distance(coordinates[i][0], coordinates[i][1], coordinates[j][0], coordinates[j][1])
                        if distance <= D:
                            infected[j] = True
                            new_infections += 1
        if new_infections == 0:
            break

    return infected

def main():
    # Read input from stdin
    N, D = map(int, sys.stdin.readline().split())
    coordinates = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]

    # Spread the virus
    infected = spread_virus(N, D, coordinates)

    # Print the result
    for i in range(N):
        if infected[i]:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()