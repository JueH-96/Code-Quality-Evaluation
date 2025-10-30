# YOUR CODE HERE
import sys

# Function to encapsulate the solution logic
def solve():
    # Read N (number of boxes) and Q (number of queries) from standard input
    # Use sys.stdin.readline for faster input processing compared to input()
    N, Q = map(int, sys.stdin.readline().split())
    
    # Read the initial colors C_1, C_2, ..., C_N for the N boxes
    C = list(map(int, sys.stdin.readline().split()))

    # Initialize the contents of the boxes. We use a list `box_contents` where
    # `box_contents[i]` stores the set of colors for box i+1 (using 0-based indexing internally).
    # A set is used because we need to count the number of *distinct* colors.
    box_contents = []
    # Each box `i` (corresponding to box number i+1) initially contains one ball of color C[i].
    # We store this single color in a set for each box.
    for i in range(N):
        # Create a set containing the single color C[i] for box i+1 (at index i)
        box_contents.append({C[i]}) 

    # Process each of the Q queries sequentially
    for _ in range(Q):
        # Read the query parameters a and b (box numbers, 1-based indexing)
        a, b = map(int, sys.stdin.readline().split())
        # Adjust a and b to 0-based indices for list access (Python uses 0-based indexing)
        a -= 1 
        b -= 1

        # Get references to the set objects representing the contents of box a and box b.
        # Using references (`set_a = box_contents[a]`) is efficient as it avoids copying entire sets.
        # Modifications to `set_a` or `set_b` might modify the sets stored in `box_contents` if they refer to the same object.
        set_a = box_contents[a]
        set_b = box_contents[b]

        # --- Optimization Check ---
        # If the source box `a` is already empty (`set_a` is empty), moving its contents has no effect on box `b`.
        # The operation requires moving balls from `a` to `b`, then emptying `a`.
        # If `a` is already empty, it remains empty. Box `b` contents are unchanged.
        # We just need to print the number of distinct colors currently in box `b`.
        if not set_a: # Check if the set `set_a` is empty. `not set()` evaluates to True.
            print(len(set_b))
            continue # Proceed to the next query without further processing

        # --- Merge Logic using "merge small into large" heuristic ---
        # The task is to move all balls (represented by their colors) from box `a` to box `b`.
        # After the operation, box `a` must be empty, and box `b` must contain the union of their previous contents.
        # To perform this efficiently, especially when sets become large, we use the "merge small into large" strategy:
        # We always iterate over the elements of the smaller set and add them to the larger set. This minimizes the total work over time.
        
        # Case 1: The set from box `a` (`set_a`) is smaller than or equal in size to the set from box `b` (`set_b`).
        if len(set_a) <= len(set_b):
            # Iterate through each distinct color `color` present in the smaller set `set_a`.
            for color in set_a:
                # Add this `color` to the larger set `set_b`.
                # Python's `set.add()` method automatically handles duplicates: 
                # if `color` is already present in `set_b`, the set remains unchanged. This is O(1) on average.
                set_b.add(color)
            
            # After transferring colors (conceptually, by adding elements to `set_b`), box `a` must become empty.
            # We assign a new, empty set object to the position `box_contents[a]`.
            # The original set object referenced by `set_a` is effectively discarded from box `a`.
            box_contents[a] = set()
            
            # Box `b`'s contents are represented by the set object `set_b`, which has been updated in-place.
            # The reference `box_contents[b]` still points to this updated set object.
            # The number of distinct colors in box `b` is now the size of this updated set `set_b`.
            print(len(set_b))

        # Case 2: The set from box `a` (`set_a`) is strictly larger than the set from box `b` (`set_b`).
        else: # len(set_a) > len(set_b)
            # Iterate through each distinct color `color` present in the smaller set `set_b`.
            for color in set_b:
                # Add this `color` to the larger set `set_a`.
                set_a.add(color)
            
            # After merging elements into `set_a`, box `b` must contain the combined set of colors.
            # The combined set is currently stored in the set object referenced by `set_a`.
            # We update the reference `box_contents[b]` to point to this set object (`set_a`).
            # This is efficient: just changing the reference, not copying elements.
            box_contents[b] = set_a 
            
            # Box `a` must become empty.
            # We assign a new, empty set object to `box_contents[a]`.
            # The original set object that `box_contents[a]` pointed to is now correctly referenced only by `box_contents[b]`.
            # The original set object that `box_contents[b]` pointed to is no longer referenced by any box list entry and might be garbage collected if not referenced elsewhere.
            box_contents[a] = set()
            
            # The number of distinct colors in box `b` is the size of the set it now points to.
            # This is the size of the set object that was originally `set_a` but now contains the union.
            # `len(set_a)` and `len(box_contents[b])` are equal at this point because `box_contents[b]` points to the same object as `set_a`. We print the size.
            print(len(box_contents[b])) 

# Call the main function `solve()` to execute the program logic
solve()