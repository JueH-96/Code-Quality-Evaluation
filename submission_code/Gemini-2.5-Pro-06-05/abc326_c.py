import sys

def solve():
    """
    Solves the problem by reading from stdin, processing, and writing to stdout.
    """
    # Read the number of gifts N and the interval length M.
    try:
        n, m = map(int, sys.stdin.readline().split())
    except (ValueError, IndexError):
        # This handles cases like empty input lines.
        # Given the problem constraints (N >= 1), this is unlikely to be triggered.
        return

    # Read the coordinates of the N gifts.
    a = list(map(int, sys.stdin.readline().split()))
    
    # The constraints state N >= 1, so if N=0 was possible, we'd handle it here.
    # if n == 0:
    #     print(0)
    #     return

    # Sort the gift coordinates. This is crucial for the two-pointer approach.
    # Time complexity: O(N log N)
    a.sort()

    # Initialize two pointers for the sliding window and a variable for the answer.
    left = 0
    max_gifts = 0

    # Iterate through the sorted coordinates with the 'right' pointer.
    # The loop runs in O(N) time as 'right' moves N times and 'left' only moves forward.
    for right in range(n):
        # The window is defined by the interval [a[left], a[left] + M).
        # If the gift a[right] is outside this interval, shrink the window
        # from the left by incrementing the 'left' pointer.
        while a[right] - a[left] >= m:
            left += 1
        
        # The number of gifts in the current valid window [a[left]...a[right]]
        # is a candidate for the maximum. Update the maximum count.
        max_gifts = max(max_gifts, right - left + 1)

    # Print the maximum number of gifts that can be acquired.
    print(max_gifts)

solve()