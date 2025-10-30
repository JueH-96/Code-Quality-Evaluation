import sys
from collections import deque

def solve():
    """
    Reads input, solves the kagamimochi problem, and prints the result.
    """
    try:
        # Read the number of mochi, N.
        n_str = sys.stdin.readline()
        if not n_str: return
        n = int(n_str)
        
        # Read the sorted list of mochi sizes, A.
        a = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Exit gracefully on malformed or empty input.
        return

    # pending_tops will store sizes of mochi designated as "tops" that are awaiting a "bottom".
    # A deque is used for efficient (O(1)) additions to the right and removals from the left.
    pending_tops = deque()
    
    # count will store the total number of kagamimochi we can make.
    count = 0

    # Iterate through each mochi in the sorted list.
    for mochi_size in a:
        # Check if the current mochi can serve as a bottom for the smallest pending top.
        # The smallest pending top is at the front of the deque (pending_tops[0]).
        if pending_tops and 2 * pending_tops[0] <= mochi_size:
            # A pair can be formed.
            # Increment the pair count.
            count += 1
            # The top mochi is now used, so remove it from the queue of pending tops.
            pending_tops.popleft()
        else:
            # The current mochi cannot be a bottom for the smallest pending top,
            # so it cannot be a bottom for any of them.
            # It must become a candidate for a top piece itself.
            pending_tops.append(mochi_size)

    # Print the maximum number of kagamimochi that can be formed.
    print(count)

solve()