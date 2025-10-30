# YOUR CODE HERE
import sys
import bisect
from collections import deque

def solve():
    # Read input values for N, M, A, B
    N, M, A, B = map(int, sys.stdin.readline().split())
    
    # Read M bad intervals [L_i, R_i]
    bad_intervals = []
    for i in range(M):
        # Store each interval as a list [L_i, R_i]
        bad_intervals.append(list(map(int, sys.stdin.readline().split())))

    # Sort bad intervals based on their start points L_i.
    # The problem statement guarantees they are already sorted and non-overlapping, 
    # but sorting ensures correctness even if input format slightly varies.
    bad_intervals.sort()

    # Function to check if a given square `sq` is "good" (i.e., not bad).
    # A square is bad if it falls within any of the bad intervals.
    # This function uses binary search on the sorted list of bad intervals for efficiency.
    def check_good(sq):
        # Squares must be within the valid range [1, N].
        if sq < 1 or sq > N:
            return False
        
        # Use binary search (bisect_left) to find the insertion point for an interval starting slightly after sq.
        # This helps locate the relevant interval potentially containing sq.
        # `bisect_left(bad_intervals, [sq + 1, 0])` finds the index `idx` such that all intervals `bad_intervals[j]` with `j < idx`
        # have `L_j <= sq`. The interval `bad_intervals[idx-1]` (if it exists) is the only candidate that might contain `sq`.
        idx = bisect.bisect_left(bad_intervals, [sq + 1, 0]) 
        
        # Check the interval immediately preceding the insertion point `idx`.
        if idx > 0:
            # Get the L and R values of the candidate interval.
            L, R = bad_intervals[idx-1]
            # If sq falls within this interval [L, R], it is a bad square.
            if L <= sq <= R:
                return False 
        
        # If sq was not found in any bad interval, it is a good square.
        return True

    # Initialize a double-ended queue (deque) for the Breadth-First Search (BFS).
    # The BFS will operate on intervals of reachable squares.
    queue = deque()
    
    # Initialize a set to keep track of intervals (start, end) that have been added to the queue.
    # This prevents processing the same interval multiple times, avoiding redundant work and potential infinite loops.
    # Intervals are stored as tuples (start, end) for hashability in the set.
    processed = set()

    # Set up the initial state for the BFS.
    # The starting square is 1. We represent the initial reachable state as the interval [1, 1].
    if check_good(1):
        # If square 1 is good (as guaranteed by constraints), add the interval [1, 1] to the queue and mark it as processed.
        queue.append((1, 1))
        processed.add((1, 1))
    else:
        # If square 1 were bad (this case shouldn't occur based on problem constraints),
        # then square N would be unreachable. Print "No" and terminate.
        print("No")
        return

    # Flag to indicate whether square N has been determined to be reachable.
    final_N_reached = False

    # Start the BFS loop. The loop continues as long as there are intervals in the queue to process.
    while queue:
        # Dequeue the next interval [s, e] from the front of the queue.
        s, e = queue.popleft()
        
        # Check if square N is contained within the current interval [s, e].
        # If N is reached, set the flag and break the loop. This serves as an early termination condition.
        if s <= N <= e:
           final_N_reached = True
           break 

        # Calculate the target range of squares potentially reachable in one step from any square in the interval [s, e].
        # The minimum reachable square is s + A (from square s).
        # The maximum reachable square is e + B (from square e), capped at N.
        target_s = s + A
        target_e = min(N, e + B) 

        # If the minimum possible next square (target_s) is already beyond N, 
        # then no further moves towards N are possible from this branch. Continue to the next iteration.
        if target_s > N: 
             continue

        # Scan the target range [target_s, target_e] to identify contiguous blocks of good squares.
        # `current_scan_s` is the starting point for the current scan within the target range.
        current_scan_s = target_s
        
        # Continue scanning as long as `current_scan_s` is within the target range.
        while current_scan_s <= target_e:
            # Check if the square `current_scan_s` is good.
            if not check_good(current_scan_s):
                # If it's a bad square, skip it and move to the next square.
                current_scan_s += 1
                continue
            
            # If `current_scan_s` is good, it marks the beginning of a contiguous block of good squares.
            good_block_s = current_scan_s
            
            # Determine the end of this contiguous block of good squares.
            # The block ends either at `target_e` or just before the start of the next bad interval.
            # Use binary search (`bisect_left`) to find the index `idx` of the first bad interval [L, R] such that L >= good_block_s.
            idx = bisect.bisect_left(bad_intervals, [good_block_s, 0]) 

            # Determine the starting square `L` of the next bad interval.
            # If no relevant bad interval exists within or after the current block, set it to N + 1 (effectively infinity).
            next_bad_start = N + 1 
            if idx < len(bad_intervals):
                 next_bad_start = bad_intervals[idx][0]

            # The good block ends at the minimum of `target_e` and `next_bad_start - 1`.
            good_block_e = min(target_e, next_bad_start - 1)

            # We have now identified a contiguous block of reachable good squares: [good_block_s, good_block_e].
            
            # Check if this interval block represents a new state space that needs exploration.
            # Convert the interval to a tuple for set membership checking.
            interval_tuple = (good_block_s, good_block_e)
            if interval_tuple not in processed:
                 # If this interval has not been encountered and added to the queue before:
                 # Add it to the queue for future processing.
                 queue.append(interval_tuple)
                 # Mark it as processed to avoid re-adding.
                 processed.add(interval_tuple)

                 # Check if square N falls within this newly discovered reachable interval.
                 if good_block_s <= N <= good_block_e:
                      # If N is reached, set the flag indicating success.
                      final_N_reached = True
                      # Break out of the inner scanning loop (while current_scan_s <= target_e).
                      break 

            # After processing the block [good_block_s, good_block_e], advance the scan pointer
            # to the square immediately following this block.
            current_scan_s = good_block_e + 1
        
        # If the `final_N_reached` flag was set within the inner loop (meaning N was found),
        # break out of the outer BFS loop as well.
        if final_N_reached:
             break

    # After the BFS loop finishes (either because the queue became empty or N was reached):
    # Check the flag `final_N_reached`.
    if final_N_reached:
        # If N was reached, print "Yes".
        print("Yes")
    else:
        # If the loop completed without reaching N, print "No".
        print("No") 

# Call the main function to solve the problem.
solve()