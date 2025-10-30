import sys

def solve():
    # Read N and Q from the first line of input
    N, Q = map(int, sys.stdin.readline().split())

    # Read the initial colors C_i from the second line
    C = list(map(int, sys.stdin.readline().split()))

    # Initialize boxes.
    # boxes[i] will be a set containing colors in box (i+1).
    # Each box initially contains one ball of color C_i.
    # We use a list of sets, where each set represents the contents of a box.
    boxes = [set() for _ in range(N)]
    for i in range(N):
        boxes[i].add(C[i])

    # A list to store the results of each query to be printed at the end
    results = []

    # Process each of the Q queries
    for _ in range(Q):
        a, b = map(int, sys.stdin.readline().split())
        
        # Adjust box numbers (a and b) to 0-based indices for array access
        idx_a = a - 1
        idx_b = b - 1

        # Get references to the set objects currently in box 'a' and box 'b'
        set_a = boxes[idx_a]
        set_b = boxes[idx_b]

        # Apply the move logic using a small-to-large merging strategy for efficiency.
        # The goal is to move all balls from box 'a' to box 'b', and make box 'a' empty.
        
        if len(set_a) == 0:
            # Case 1: Box 'a' is already empty.
            # No balls to move. Box 'b' content remains unchanged. Box 'a' remains empty.
            # No operations needed on sets.
            pass 
        elif len(set_b) == 0:
            # Case 2: Box 'b' is empty.
            # All balls from box 'a' simply become the contents of box 'b'.
            # Box 'a' becomes empty (by assigning the now-empty original set_b to it).
            boxes[idx_b] = set_a  # Box 'b' now points to the set object that was 'set_a'
            boxes[idx_a] = set_b  # Box 'a' now points to the set object that was 'set_b' (which is empty)
        else:
            # Case 3: Both boxes contain balls. Perform the merge.
            # To optimize, we iterate over the smaller set and add its elements to the larger set.
            # After merging, the combined set (the one that received elements) will be the content of box 'b'.
            # The other set (the one whose elements were transferred) will be cleared and become the content of box 'a'.
            
            if len(set_a) > len(set_b):
                # If set_a is larger, iterate over elements of set_b and add them to set_a.
                # Then set_a (now combined) becomes the new content of box 'b'.
                # Set_b is cleared and becomes the new content of box 'a'.
                for color in set_b:
                    set_a.add(color)
                set_b.clear() # Clear the set that was originally set_b
                
                boxes[idx_b] = set_a # Combined set (original set_a) is now content of box b
                boxes[idx_a] = set_b # Cleared set (original set_b) is now content of box a
            else:
                # If set_b is larger or equal, iterate over elements of set_a and add them to set_b.
                # Then set_b (now combined) remains the content of box 'b'.
                # Set_a is cleared and becomes the new content of box 'a'.
                for color in set_a:
                    set_b.add(color)
                set_a.clear() # Clear the set that was originally set_a
                
                boxes[idx_b] = set_b # Combined set (original set_b) is still content of box b
                boxes[idx_a] = set_a # Cleared set (original set_a) is now content of box a

        # After the merge operation, the number of different colors in box b
        # is simply the size of the set now associated with box b.
        results.append(str(len(boxes[idx_b])))

    # Print all collected results, each on a new line.
    sys.stdout.write("
".join(results) + "
")

# Call the solve function to execute the program
solve()