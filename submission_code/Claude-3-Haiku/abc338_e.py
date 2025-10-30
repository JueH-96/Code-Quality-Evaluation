import sys

def check_intersection(N, chords):
    # Create a list to store the endpoints of the chords
    endpoints = []
    for a, b in chords:
        endpoints.append((a, True))  # True indicates the start of a chord
        endpoints.append((b, False))  # False indicates the end of a chord

    # Sort the endpoints by their values
    endpoints.sort()

    # Keep track of the number of open chords
    open_chords = 0

    for endpoint, is_start in endpoints:
        if is_start:
            open_chords += 1
            if open_chords > 1:
                return "Yes"
        else:
            open_chords -= 1

    return "No"

# Read the input from stdin
N = int(sys.stdin.readline().strip())
chords = []
for _ in range(N):
    a, b = map(int, sys.stdin.readline().strip().split())
    chords.append((a, b))

# Check for intersection and print the result
print(check_intersection(N, chords))