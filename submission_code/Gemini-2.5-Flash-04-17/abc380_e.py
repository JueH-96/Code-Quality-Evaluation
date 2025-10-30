import sys
# Using sortedcontainers library for SortedDict
from sortedcontainers import SortedDict

# Recommended for competitive programming in Python for speed
input = sys.stdin.readline

def solve():
    # Read N (number of cells) and Q (number of queries)
    N, Q = map(int, input().split())

    # segments: SortedDict[start_index] = (end_index, color)
    # This dictionary stores the contiguous segments of cells that have the same color.
    # Keys are the starting indices of these segments, maintained in sorted order by SortedDict.
    # Example: segments = {1: (3, 5), 4: (4, 2), 5: (7, 5)} represents:
    # cells 1 through 3 have color 5, cell 4 has color 2, and cells 5 through 7 have color 5.
    segments = SortedDict()

    # color_counts: Dict[color] = count
    # This dictionary stores the total number of cells currently painted with each color.
    color_counts = {}

    # Initial state: cell i is painted with color i.
    # This means initially we have N segments, each consisting of a single cell.
    # Initialize the segments dictionary and color counts.
    for i in range(1, N + 1):
        segments[i] = (i, i)
        color_counts[i] = 1

    # Process Q queries
    for _ in range(Q):
        # Read query parameters
        query = list(map(int, input().split()))
        query_type = query[0]

        if query_type == 1:
            # Type 1 query: repaint reachable component starting from cell x to color c.
            x, c = query[1], query[2]

            # Find the segment that contains cell x.
            # The keys of the `segments` SortedDict are the start indices of the segments, and they are sorted.
            # We need to find the largest start index L in `segments` such that L <= x.
            # `segments.keys()` provides a sorted view of the start indices.
            # `bisect_right(x)` finds the index `i` in `segments.keys()` such that `segments.keys()[i-1] <= x < segments.keys()[i]`.
            # Therefore, `segments.keys()[i-1]` is the start index of the segment containing cell x.
            keys = segments.keys()
            idx = keys.bisect_right(x) - 1 # Get the index of the largest key (start index) <= x
            L, R, color_x = keys[idx], segments[keys[idx]][0], segments[keys[idx]][1]

            # If the current color of the segment containing x is already the target color c,
            # the repaint operation has no effect. Skip the rest of the query processing.
            if color_x == c:
                continue

            # The segment [L, R] with color color_x is being repainted to color c.
            # Update color counts: subtract the size of the segment from the old color's count
            # and add it to the new color's count.
            segment_size = R - L + 1
            color_counts[color_x] -= segment_size
            # Ensure the new color entry exists in color_counts before adding. If color c
            # wasn't present before, its count starts at 0.
            color_counts[c] = color_counts.get(c, 0) + segment_size

            # Remove the old segment [L, R] with color color_x from the dictionary.
            # This segment is conceptually replaced by a new segment (possibly merged).
            del segments[L]

            # Determine the potential new boundaries for the segment after repainting and merging.
            current_L = L # Start with the original left boundary
            current_R = R # Start with the original right boundary

            # Check for a potential merge with the segment immediately to the left.
            # A merge is possible if there is a segment ending exactly at L - 1 and that segment
            # already has the new target color c.
            # The start key of this potential left segment is the largest key strictly less than L.
            # After deleting segments[L], `segments.keys().bisect_left(L)` gives the index in the
            # current sorted keys where L would be inserted. The key at the index before this
            # (`index - 1`) is the one immediately preceding the position of L.
            idx_prev = segments.keys().bisect_left(L)
            if idx_prev > 0: # Check if there is at least one key before L's insertion point
                L_prev = segments.keys()[idx_prev - 1] # Get the start index of the left segment
                R_prev, color_prev = segments[L_prev] # Get its end index and color
                # Check if the left segment is contiguous with the old segment [L, R] (i.e., ends at L-1)
                # AND if it has the target color c.
                if R_prev == L - 1 and color_prev == c:
                    # Perform left merge: the new segment's left boundary becomes the start of the left segment.
                    current_L = L_prev
                    # We don't delete segments[L_prev] here; it will be updated later.

            # Check for a potential merge with the segment immediately to the right.
            # A merge is possible if there is a segment starting exactly at R + 1 and that segment
            # already has the new target color c.
            L_next = R + 1
            # Check if R + 1 is a valid cell index (within [1, N]) AND if there is a segment
            # in our `segments` dictionary that starts exactly at R + 1.
            if L_next <= N and L_next in segments:
                R_next, color_next = segments[L_next] # Get its end index and color
                # Check if the right segment has the target color c.
                if color_next == c:
                    # Perform right merge: the new segment's right boundary becomes the end of the right segment.
                    current_R = R_next
                    # The segment starting at L_next is now part of the larger, merged segment. Delete it.
                    del segments[L_next]

            # Insert or update the segments dictionary with the new (potentially merged) segment.
            # If a left merge happened, current_L is L_prev. `segments[L_prev]` is updated.
            # If only a right merge happened or no merge, current_L is L. `segments[L]` is inserted
            # (it was previously deleted).
            segments[current_L] = (current_R, c)

        elif query_type == 2:
            # Type 2 query: print the number of cells painted with color c.
            c = query[1]
            # Retrieve the count for color c from the color_counts dictionary.
            # Use get(c, 0) to handle cases where color c might not be present in the dictionary
            # (meaning its count is 0).
            print(color_counts.get(c, 0))

# Execute the main solve function
solve()