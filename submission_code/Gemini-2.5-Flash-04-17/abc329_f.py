import sys

# Increase recursion depth if necessary (not strictly needed for this iterative solution)
# sys.setrecursionlimit(2000)

# Read N, Q
N, Q = map(int, sys.stdin.readline().split())

# Read initial colors C (using 1-based indexing for colors in problem, list C is 0-indexed)
C = list(map(int, sys.stdin.readline().split()))

# Initialize boxes
# boxes is a list of sets. boxes[i] stores the colors in box i+1.
# We use a list of N sets, indexed 0 to N-1.
# Each box initially contains one ball of color C[i].
boxes = [set() for _ in range(N)]
for i in range(N):
    boxes[i].add(C[i]) # Initial color C[i] goes into box i (0-indexed)

# Process queries
for _ in range(Q):
    # Read query (a, b)
    a, b = map(int, sys.stdin.readline().split())
    # Adjust to 0-based indices
    idx_a = a - 1
    idx_b = b - 1

    # Get the sets from the boxes using their current references
    source_set = boxes[idx_a]
    dest_set = boxes[idx_b]

    # Handle empty source box case: nothing moves
    if not source_set:
        # Box a is empty, no balls to move. Box b remains as is.
        print(len(dest_set))
        # The references in boxes remain unchanged.
        # boxes[idx_a] is already empty.
        # boxes[idx_b] is unchanged.
        continue

    # Smaller-to-larger merging for efficiency
    # We want to move all balls (colors) from box a (source_set) to box b (dest_set).
    # The result of merging source_set into dest_set will go into boxes[idx_b].
    # Iterate over the smaller set and add its elements to the larger one.
    if len(source_set) > len(dest_set):
        # Source set is larger, destination set is smaller.
        # To maintain the smaller-to-larger principle, we iterate through the
        # smaller set (dest_set) and add its elements to the larger set (source_set).
        # The modified source_set will then represent the merged set.
        set_to_iterate = dest_set
        set_to_add_to = source_set
        # The final merged set will be set_to_add_to (which is the original source_set)
        # after adding elements from set_to_iterate (original dest_set).
        result_set_for_box_b = set_to_add_to
    else: # len(source_set) <= len(dest_set)
        # Source set is smaller or equal, destination set is larger or equal.
        # Iterate through the smaller/equal set (source_set) and add its elements
        # to the larger/equal set (dest_set). The modified dest_set will
        # represent the merged set.
        set_to_iterate = source_set
        set_to_add_to = dest_set
        # The final merged set will be set_to_add_to (which is the original dest_set)
        # after adding elements from set_to_iterate (original source_set).
        result_set_for_box_b = set_to_add_to

    # Perform the merge iteration
    # This modifies the set object referenced by set_to_add_to in place.
    for color in set_to_iterate:
        set_to_add_to.add(color)

    # Assign the result set to the destination box
    # The destination box (idx_b) now refers to the merged set.
    boxes[idx_b] = result_set_for_box_b

    # After moving all balls from box a, box a becomes empty.
    # Assign a *new* empty set to boxes[idx_a].
    # This is important to ensure that box idx_a is truly empty and
    # does not accidentally share a reference with the set now in box idx_b.
    boxes[idx_a] = set()

    # Print the number of distinct colors in box b (which now holds the merged set)
    print(len(boxes[idx_b]))