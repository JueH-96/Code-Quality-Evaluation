import sys
from bisect import bisect_left

def solve():
    """
    This function encapsulates the entire logic for solving the problem.
    It reads from stdin, processes, and writes to stdout.
    """
    try:
        # Read problem parameters from standard input
        N = int(sys.stdin.readline())
        A = list(map(int, sys.stdin.readline().split()))
        B = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Handle potential empty input or format errors gracefully
        return

    # The core of the problem is a matching problem. For a valid matching to exist
    # between N toys and N boxes, it's necessary and sufficient that when both
    # are sorted by size, the i-th toy is no larger than the i-th box.
    # We sort the initial toy and box lists. This is O(N log N).
    A.sort()
    B.sort()

    def can_place_all(x):
        """
        Checks if it's possible to place all N toys given an additional box of size x.
        The condition for success is A_sorted[i] <= AllBoxes_sorted[i] for all i.
        
        Args:
            x (int): The size of the new box.
            
        Returns:
            bool: True if placement is possible, False otherwise.
        """
        # A and B are pre-sorted. To get the sorted list of all N boxes,
        # we find the insertion point of x in B.
        # bisect_left finds the index k where x can be inserted while maintaining order.
        k = bisect_left(B, x)
        
        # Check the placement condition efficiently without creating a new list.
        # This is equivalent to checking A[i] <= all_boxes[i] for all i, where
        # all_boxes is the sorted list of B and x.
        
        # Check toys A[0]..A[k-1] against boxes B[0]..B[k-1]
        for i in range(k):
            if A[i] > B[i]:
                return False
        
        # Check toy A[k] against the new box x
        if A[k] > x:
            return False
            
        # Check toys A[k+1]..A[N-1] against boxes B[k]..B[N-2]
        # Note: toy A[i] corresponds to box B[i-1] in this segment.
        for i in range(k + 1, N):
            if A[i] > B[i-1]:
                return False
                
        return True

    # The function `can_place_all(x)` is monotonic. This allows us to use 
    # binary search to find the minimum `x` for which `can_place_all(x)` is true.

    # Set up the binary search range.
    low = 1
    high = 2 * 10**9 # A safe, large upper bound
    min_x = -1

    # Perform binary search
    while low <= high:
        mid = (low + high) // 2
        
        if can_place_all(mid):
            # If `mid` works, it's a potential answer. We store it and try
            # to find an even smaller `x` in the lower half of the range.
            min_x = mid
            high = mid - 1
        else:
            # If `mid` doesn't work, we need a larger box. Search in the upper half.
            low = mid + 1
            
    print(min_x)

# The `if __name__ == "__main__":` block ensures that `solve()` is called
# only when the script is executed directly.
if __name__ == "__main__":
    solve()