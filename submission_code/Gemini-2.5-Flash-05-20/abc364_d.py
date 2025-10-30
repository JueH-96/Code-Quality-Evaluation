import sys
import bisect

def solve():
    # Read N and Q from the first line
    N, Q = map(int, sys.stdin.readline().split())

    # Read the coordinates of points A_i
    a = list(map(int, sys.stdin.readline().split()))
    # Sort the A_i coordinates. This is crucial for efficient range queries
    # using binary search (bisect_left and bisect_right).
    a.sort()

    # A list to store the answers for each query
    results = []

    # Process each of the Q queries
    for _ in range(Q):
        # Read the B_j coordinate and k_j (the k-th closest point)
        b_j, k_j = map(int, sys.stdin.readline().split())

        # Define a helper function to count how many A_i points are within a certain distance 'val' from B_j.
        # This is equivalent to counting A_i such that |A_i - b_j| <= val.
        # This inequality can be rewritten as: -val <= A_i - b_j <= val
        # Which further simplifies to: b_j - val <= A_i <= b_j + val
        def count_a_within_distance(val):
            # Find the index where an element equal to or greater than (b_j - val) would be inserted.
            # This marks the left boundary of our search range in the sorted 'a' list.
            lower_bound_for_a = b_j - val
            idx_left = bisect.bisect_left(a, lower_bound_for_a)

            # Find the index where an element greater than (b_j + val) would be inserted.
            # This marks the right boundary of our search range in the sorted 'a' list.
            upper_bound_for_a = b_j + val
            idx_right = bisect.bisect_right(a, upper_bound_for_a)

            # The number of elements within the desired range is the difference between these two indices.
            return idx_right - idx_left

        # Perform a binary search on the possible values of the distance.
        # The distance can range from 0 (if b_j is exactly an a_i)
        # to 2 * 10^8 (e.g., a_i = -10^8, b_j = 10^8, distance = 2 * 10^8).
        low = 0
        high = 2 * 10**8 
        
        # 'ans' will store the smallest distance 'D' found so far for which
        # at least k_j points are within that distance.
        # Initialize it to the maximum possible distance.
        ans = high 

        # Standard binary search loop to find the minimum 'ans' that satisfies the condition.
        while low <= high:
            mid = (low + high) // 2
            
            # Check if 'mid' distance satisfies the condition (at least k_j points are within this distance).
            if count_a_within_distance(mid) >= k_j:
                # If yes, 'mid' is a potential answer. We record it and try to find an even smaller distance
                # by searching in the lower half of the current range.
                ans = mid
                high = mid - 1
            else:
                # If no, 'mid' is too small; we need a larger distance to include k_j points.
                # So, we search in the upper half of the current range.
                low = mid + 1
        
        # Add the found k_j-th closest distance to our results list
        results.append(ans)

    # Print all collected results, each on a new line.
    # Using sys.stdout.write and '
'.join for potentially faster output with many lines.
    sys.stdout.write('
'.join(map(str, results)) + '
')

# Call the main solve function to execute the program
solve()