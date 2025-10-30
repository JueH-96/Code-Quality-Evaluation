import sys

def solve():
    # Read N and M from the first line
    N, M = map(int, sys.stdin.readline().split())

    # Read the list of gift coordinates A
    A = list(map(int, sys.stdin.readline().split()))

    # Sort the gift coordinates. This is essential for the two-pointer (sliding window) approach.
    # Time complexity of sorting is O(N log N).
    A.sort()

    # Initialize variables for the sliding window
    max_gifts = 0  # Stores the maximum number of gifts found so far
    left = 0       # Left pointer of the current window [A[left], ..., A[right]]

    # Iterate with the right pointer from the beginning to the end of the sorted array.
    # The right pointer marks the current gift being considered as the rightmost in a potential optimal interval.
    # The loop runs N times.
    for right in range(N):
        # This while loop shrinks the window from the left if the span of gifts
        # A[right] - A[left] is too large (M or more).
        # The condition A[right] - A[left] >= M implies that A[left] and A[right]
        # cannot both be simultaneously included in a half-open interval [x, x+M).
        # We must advance 'left' to ensure that the current window A[left...right]
        # has a span strictly less than M, so that all gifts in it can be covered.
        # The 'left' pointer only moves forward, so in total it traverses the array at most once.
        # This makes the inner while loop's total complexity O(N) over all 'right' iterations.
        while A[right] - A[left] >= M:
            left += 1
        
        # At this point, the span A[right] - A[left] is guaranteed to be less than M.
        # This means all gifts from A[left] to A[right] can be covered by a single interval
        # of length M (e.g., by choosing x = A[left], then the interval [A[left], A[left] + M)
        # includes all gifts from A[left] up to A[right] because A[right] < A[left] + M).
        
        # Calculate the number of gifts in the current valid window.
        current_gifts = right - left + 1
        
        # Update the overall maximum number of gifts found.
        if current_gifts > max_gifts:
            max_gifts = current_gifts
            
    # Print the final maximum number of gifts.
    print(max_gifts)

# Call the solve function to execute the program logic.
solve()