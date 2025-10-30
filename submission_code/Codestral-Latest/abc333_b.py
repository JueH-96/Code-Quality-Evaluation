# YOUR CODE HERE
import sys

def distance_between_points(p1, p2):
    # Define the distances between adjacent points in the pentagon
    distances = {
        ('A', 'B'): 1, ('B', 'A'): 1,
        ('B', 'C'): 1, ('C', 'B'): 1,
        ('C', 'D'): 1, ('D', 'C'): 1,
        ('D', 'E'): 1, ('E', 'D'): 1,
        ('E', 'A'): 1, ('A', 'E'): 1,
        ('A', 'C'): 2, ('C', 'A'): 2,
        ('B', 'D'): 2, ('D', 'B'): 2,
        ('C', 'E'): 2, ('E', 'C'): 2,
        ('D', 'A'): 2, ('A', 'D'): 2,
        ('E', 'B'): 2, ('B', 'E'): 2
    }

    # If the points are the same, the distance is 0
    if p1 == p2:
        return 0

    # Return the distance between the points
    return distances.get((p1, p2), float('inf'))

def main():
    # Read input from stdin
    input = sys.stdin.read().strip().split()
    S_1S_2 = input[0]
    T_1T_2 = input[1]

    # Extract points
    S_1, S_2 = S_1S_2[0], S_1S_2[1]
    T_1, T_2 = T_1T_2[0], T_1T_2[1]

    # Calculate distances
    distance_S = distance_between_points(S_1, S_2)
    distance_T = distance_between_points(T_1, T_2)

    # Compare distances and print result
    if distance_S == distance_T:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()