import sys
from collections import defaultdict

def max_gifts(N, M, A):
    # Create a dictionary to count the gifts at each coordinate
    gift_count = defaultdict(int)
    for a in A:
        gift_count[a] += 1

    # Extract the unique coordinates and sort them
    coordinates = sorted(gift_count.keys())

    # Initialize variables to track the maximum number of gifts
    max_gifts = 0
    current_gifts = 0
    start = 0

    # Use a sliding window approach to find the maximum number of gifts
    for end in range(len(coordinates)):
        # Add the gifts at the end coordinate to the current window
        current_gifts += gift_count[coordinates[end]]

        # While the window size is greater than M, remove gifts from the start
        while coordinates[end] - coordinates[start] >= M:
            current_gifts -= gift_count[coordinates[start]]
            start += 1

        # Update the maximum number of gifts
        max_gifts = max(max_gifts, current_gifts)

    return max_gifts

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

# Calculate and print the result
result = max_gifts(N, M, A)
print(result)