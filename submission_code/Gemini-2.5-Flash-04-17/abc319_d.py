# YOUR CODE HERE
import sys

# Function to check if sentence fits in M lines with window width W
def can_fit(W, N, M, L):
    """
    Checks if the sentence with word lengths L can fit into M or fewer lines
    with a window width of W.
    """
    # Note: We don't need an explicit check if length > W for any word inside the loop,
    # because the binary search lower bound ensures W >= max(L).

    lines_count = 1
    current_line_width = 0

    for length in L:
        # If this is the first word on the current line
        if current_line_width == 0:
            current_line_width = length
        # If adding this word (with a space) fits on the current line
        # Required width: current_line_width (from previous words) + 1 (space) + length (current word)
        elif current_line_width + 1 + length <= W:
            current_line_width += 1 + length
        # If it doesn't fit on the current line, start a new line
        else:
            lines_count += 1
            # If we exceed the maximum allowed lines, return False immediately
            if lines_count > M:
                return False
            # Place the current word at the beginning of the new line
            current_line_width = length

    # If we processed all words without exceeding M lines, it's possible
    return True

# Read input
# sys.stdin = open('input.txt', 'r') # Uncomment for local testing with input.txt
N, M = map(int, sys.stdin.readline().split())
L = list(map(int, sys.stdin.readline().split()))

# Binary search for the minimum possible window width
# The minimum possible width (lower bound for binary search) is the width of the widest single word.
# Any width smaller than max(L) cannot even fit the widest word on a line by itself.
low = max(L)

# The maximum possible useful width (upper bound for binary search) is one that allows fitting all words in a single line.
# The width required to fit all N words in a single line is the sum of their lengths plus N-1 spaces (if N > 1).
# This total width is sum(L) + max(0, N-1).
# A simpler and safe upper bound is sum(L) + N. If W = sum(L) + N,
# the required space for one line (sum(L) + max(0, N-1)) will always be less than or equal to W for N >= 1.
# Using sum(L) + N is simpler and sufficient.
high = sum(L) + N

# Initialize the answer. We know 'high' is a working width (it fits in 1 line, and M >= 1).
# We want the minimum width, so we initialize ans to a value that is guaranteed to work
# and then try to find a smaller working width.
ans = high

# Perform binary search to find the minimum working width
while low <= high:
    # Calculate the midpoint width to test. Using (low + high) // 2 is safe in Python
    # as integers have arbitrary precision.
    mid = (low + high) // 2

    # Check if it's possible to fit the sentence in M or fewer lines with width 'mid'
    if can_fit(mid, N, M, L):
        # If it fits, 'mid' is a possible minimum answer.
        # Store it as the current best answer and try a smaller width
        # by searching in the left half [low, mid-1].
        ans = mid
        high = mid - 1
    else:
        # If it doesn't fit, 'mid' is too small.
        # We need a larger width, so search in the right half [mid+1, high].
        low = mid + 1

# After the loop finishes (low > high), 'ans' holds the minimum width found that works.
# Print the result
print(ans)