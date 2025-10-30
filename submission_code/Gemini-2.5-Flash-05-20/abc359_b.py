import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # positions[color_val] will store a list of 0-indexed positions
    # where people wearing clothes of color_val are found.
    # Since each color appears exactly twice, each list will have two elements.
    positions = [[] for _ in range(N + 1)]

    # Populate the positions list
    for i in range(2 * N):
        color = A[i]
        positions[color].append(i)

    count_satisfying_colors = 0

    # Iterate through each color from 1 to N
    for color_val in range(1, N + 1):
        # Get the two indices where this color appears
        idx1, idx2 = positions[color_val]

        # The problem statement uses 1-based indexing for positions,
        # but our array A is 0-indexed.
        # If two people are at 0-indexed positions `p1` and `p2` (where `p1 < p2`),
        # the people between them are at positions `p1+1, p1+2, ..., p2-1`.
        # The number of people between them is (p2 - 1) - (p1 + 1) + 1 = p2 - p1 - 1.
        # We need this count to be exactly 1.
        # So, p2 - p1 - 1 = 1  =>  p2 - p1 = 2.
        # This means the difference between their 0-indexed array positions must be 2.

        if idx2 - idx1 == 2:
            count_satisfying_colors += 1

    print(count_satisfying_colors)

# Call the solve function to execute the program
solve()