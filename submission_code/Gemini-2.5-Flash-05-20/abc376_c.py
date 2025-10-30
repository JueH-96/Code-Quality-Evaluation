import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Sort toys once, as their sizes are fixed and this ordering is used in check(x).
    A.sort()

    def check(x):
        """
        Checks if all N toys can be placed into N boxes (N-1 existing + 1 new of size x).
        """
        # Create a list of all N available boxes.
        all_boxes = B + [x]
        
        # Sort all boxes in ascending order.
        # This is O(N log N) and is the dominant part of check(x) if A is pre-sorted.
        all_boxes.sort()

        # Greedy matching strategy:
        # Iterate through the sorted toys (smallest to largest) and try to fit each
        # into the smallest available box that can accommodate it. This leaves larger
        # boxes for larger toys, maximizing the chances of a successful placement.
        
        box_ptr = 0 # Pointer for the sorted all_boxes list
        
        for toy_size in A:
            # Advance box_ptr to find the first box that is large enough for toy_size.
            # We are looking for all_boxes[box_ptr] >= toy_size.
            while box_ptr < N and all_boxes[box_ptr] < toy_size:
                box_ptr += 1
            
            # If box_ptr reaches N, it means there are no more boxes left or
            # all remaining boxes are too small for the current toy_size.
            if box_ptr == N:
                return False # Cannot place all toys
            
            # A suitable box (all_boxes[box_ptr]) has been found for toy_size.
            # Mark this box as used by advancing the pointer to the next box.
            box_ptr += 1
            
        return True # All toys successfully placed

    # Binary search for the minimum x.
    # The search range for x is [1, 2 * 10^9].
    # Minimum possible x is 1.
    # Maximum A_i is 10^9, so x could potentially need to be 10^9.
    # A safe upper bound for x, considering values of A_i up to 10^9, is 2 * 10^9.
    low = 1
    high = 2 * 10**9 # Maximum possible value for x.
    ans = -1         # Stores the minimum x found that allows placement. Initialized to -1 if no solution.

    while low <= high:
        mid = low + (high - low) // 2
        if check(mid):
            # If check(mid) is True, mid is a possible answer.
            # We store it and try to find a smaller x in the lower half.
            ans = mid
            high = mid - 1
        else:
            # If check(mid) is False, mid is too small.
            # We need a larger x, so search in the upper half.
            low = mid + 1

    # Print the minimum x found, or -1 if no solution exists.
    print(ans)

# Call the solve function to run the program.
solve()