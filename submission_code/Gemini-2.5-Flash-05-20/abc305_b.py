import sys

def solve():
    # Read the two points p and q from standard input
    p, q = sys.stdin.readline().split()

    # Define the coordinates of each point, assuming A is at position 0.
    # The coordinates are cumulative distances from point A.
    # A: 0
    # B: A + 3 = 3
    # C: B + 1 = 4
    # D: C + 4 = 8
    # E: D + 1 = 9
    # F: E + 5 = 14
    # G: F + 9 = 23
    coordinates = {
        'A': 0,
        'B': 3,
        'C': 4,
        'D': 8,
        'E': 9,
        'F': 14,
        'G': 23
    }

    # Get the numerical coordinate for point p
    coord_p = coordinates[p]
    # Get the numerical coordinate for point q
    coord_q = coordinates[q]

    # Calculate the distance between p and q, which is the absolute
    # difference of their coordinates on the line.
    distance = abs(coord_p - coord_q)

    # Print the calculated distance to standard output
    print(distance)

# Call the solve function to execute the program
solve()