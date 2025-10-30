import sys

def min_window_width(N, M, L):
    """
    Calculate the minimum possible width of the window.

    Args:
    N (int): The number of words.
    M (int): The maximum number of lines.
    L (list): A list of word widths.

    Returns:
    int: The minimum possible width of the window.
    """
    def check(W):
        """
        Check if it's possible to fit the sentence into M lines with window width W.

        Args:
        W (int): The window width.

        Returns:
        bool: True if it's possible, False otherwise.
        """
        lines = 1
        width = 0
        for length in L:
            if width + length + (width > 0) > W:
                lines += 1
                width = 0
            if lines > M:
                return False
            width += length + (width > 0)
        return True

    # Binary search for the minimum window width
    left, right = max(L), sum(L) + N - 1
    while left < right:
        mid = (left + right) // 2
        if check(mid):
            right = mid
        else:
            left = mid + 1
    return left

# Read input from stdin
N, M = map(int, sys.stdin.readline().split())
L = list(map(int, sys.stdin.readline().split()))

# Calculate and print the minimum window width
print(min_window_width(N, M, L))