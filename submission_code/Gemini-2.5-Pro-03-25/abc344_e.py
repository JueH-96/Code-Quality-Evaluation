# YOUR CODE HERE
import sys

# Function to handle the logic for processing queries and maintaining the sequence
def solve():
    # Reading input using sys.stdin.readline for potentially faster I/O
    N = int(sys.stdin.readline())
    # Read the initial sequence A
    A = list(map(int, sys.stdin.readline().split()))
    # Read the number of queries
    Q = int(sys.stdin.readline())

    # Data structures to represent the doubly linked list using dictionaries:
    # next_node[val] stores the value of the element immediately following 'val'.
    # prev_node[val] stores the value of the element immediately preceding 'val'.
    # Using dictionaries allows for O(1) average time complexity for lookups, insertions, and deletions.
    next_node = {}
    prev_node = {}

    # Initialize the doubly linked list based on the initial sequence A
    # The constraint N >= 1 guarantees that the initial list is not empty.
    
    current_head = None # Initialize head reference
    if N > 0:
        current_head = A[0] # The first element of A is the initial head
        prev_node[A[0]] = None # The head has no predecessor
        
        # Iterate through the initial list to set up forward (next) and backward (prev) links
        for i in range(N - 1):
            next_node[A[i]] = A[i+1] # Link current element A[i] to the next A[i+1]
            prev_node[A[i+1]] = A[i] # Link next element A[i+1] back to the current A[i]
        
        # The last element of the initial sequence, A[N-1], has no successor.
        # Set its next pointer to None. This needs to be explicitly set.
        # This line correctly handles the case N=1 as well, where A[N-1] is A[0].
        next_node[A[N-1]] = None 

    # Process each of the Q queries
    for _ in range(Q):
        # Read the query line and parse its components
        query_line = sys.stdin.readline().split()
        query_type = int(query_line[0])

        if query_type == 1:
            # Query Type 1: Insert element 'y' immediately after element 'x'
            x = int(query_line[1])
            y = int(query_line[2])
            
            # It's guaranteed that 'x' exists in the sequence at this point.
            # Find the element that is currently after 'x'. This could be None if 'x' is the tail.
            original_next_x = next_node.get(x) 

            # Update pointers to insert 'y' between 'x' and 'original_next_x':
            # 1. 'x's next pointer should now point to 'y'.
            next_node[x] = y
            # 2. 'y's previous pointer should point back to 'x'.
            prev_node[y] = x
            # 3. 'y's next pointer should point to the element that was originally after 'x'.
            next_node[y] = original_next_x

            # 4. If there was an element originally after 'x' (i.e., original_next_x is not None),
            #    update its previous pointer to point to the newly inserted 'y'.
            if original_next_x is not None:
                prev_node[original_next_x] = y

        elif query_type == 2:
            # Query Type 2: Remove element 'x' from the sequence
            x = int(query_line[1])

            # It's guaranteed that 'x' exists in the sequence at this point.
            # Get the elements immediately before and after 'x'. These could be None if x is head/tail.
            prev_x = prev_node.get(x) 
            next_x = next_node.get(x) 

            # Update the 'next' pointer of the element before 'x' (if 'x' had a predecessor)
            if prev_x is not None:
                # Make the predecessor point to the successor of 'x', effectively bypassing 'x'.
                next_node[prev_x] = next_x
            else:
                # If 'x' had no predecessor (prev_x is None), it means 'x' was the head.
                # The new head of the sequence becomes the element that was after 'x'.
                # The constraint guarantees the list is never empty, so if x is the only element,
                # next_x will be None, and current_head will become None, but this state won't persist.
                current_head = next_x

            # Update the 'prev' pointer of the element after 'x' (if 'x' had a successor)
            if next_x is not None:
                # Make the successor point back to the predecessor of 'x', effectively bypassing 'x'.
                prev_node[next_x] = prev_x

            # Remove 'x' completely by deleting its entries from both dictionaries.
            # This is safe as 'x' is guaranteed to exist as a key in both dictionaries at this point.
            del prev_node[x]
            del next_node[x]

    # After processing all queries, construct the final sequence for output
    result = []
    curr = current_head # Start traversal from the current head of the sequence
    
    # Traverse the linked list using the 'next_node' pointers until the end is reached (curr becomes None)
    while curr is not None:
        result.append(str(curr)) # Add the value of the current element to the result list as a string
        # Move to the next element. Using .get() ensures graceful handling:
        # If curr is not the tail, it gets the next value.
        # If curr is the tail, next_node.get(curr) returns None, which terminates the loop.
        curr = next_node.get(curr) 

    # Print the final sequence with elements separated by spaces
    print(" ".join(result))

# Check if the script is being run directly (not imported) and execute the main logic
if __name__ == '__main__':
    solve()