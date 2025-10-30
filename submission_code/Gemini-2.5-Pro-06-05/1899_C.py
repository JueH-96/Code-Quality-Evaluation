# YOUR CODE HERE
import sys

def solve():
    """
    This function contains the logic to solve a single test case.
    """
    try:
        # Read the size of the array
        n_str = sys.stdin.readline()
        if not n_str: return
        n = int(n_str)
        
        # Read the array elements
        a = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        # Handle potential empty lines or malformed input at the end of a file
        return

    # max_so_far stores the overall maximum sum found for any valid subarray.
    # Initialize with the first element's value, as a single element is a valid subarray.
    max_so_far = a[0]
    
    # current_max stores the maximum sum of a valid subarray ending at the current index.
    current_max = a[0]

    for i in range(1, n):
        # Check if the parity of the current element a[i] is different from the previous one a[i-1].
        # In Python, `x % 2` gives consistent results for parity checks across positive and negative integers.
        # The condition `(x % 2) != (y % 2)` reliably checks if parities are different.
        if (a[i] % 2) != (a[i-1] % 2):
            # Parities are different, so we can extend the current alternating subarray.
            # We choose the better of two options:
            # 1. Start a new subarray with just a[i].
            # 2. Extend the previous best valid subarray sum with a[i].
            current_max = max(a[i], current_max + a[i])
        else:
            # Parities are the same, breaking the alternating sequence.
            # We must start a new subarray, which can only be [a[i]].
            current_max = a[i]
        
        # Update the overall maximum sum found so far.
        max_so_far = max(max_so_far, current_max)

    print(max_so_far)


def main():
    """
    Main function to handle multiple test cases.
    """
    try:
        # Read the number of test cases.
        t = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # Handle case of empty input for the number of test cases.
        t = 0
    
    for _ in range(t):
        solve()

if __name__ == "__main__":
    main()