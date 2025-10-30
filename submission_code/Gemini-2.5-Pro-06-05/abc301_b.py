import sys

# YOUR CODE HERE
def solve():
    """
    Reads a sequence of integers and inserts numbers between adjacent elements
    until their absolute difference is 1, according to the specified procedure.
    """
    
    # Read N, not strictly necessary for the logic, but it's in the input.
    try:
        _ = int(sys.stdin.readline())
        # Read the initial sequence.
        A = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        # Handle cases with malformed input, though problem constraints suggest this won't happen.
        return

    # Loop until no more changes are made in a full pass.
    while True:
        # Flag to track if an insertion was made in the current pass.
        made_change = False
        
        # Build a new list `new_A` in each iteration.
        # Start it with the first element of the current list `A`.
        new_A = [A[0]]
        
        # Iterate through adjacent elements of `A`.
        for i in range(len(A) - 1):
            curr = A[i]
            next_val = A[i+1]
            
            # If the absolute difference is greater than 1, we need to insert numbers.
            if abs(curr - next_val) > 1:
                made_change = True
                
                # If the numbers are increasing (e.g., 2 to 5).
                if curr < next_val:
                    # Insert the numbers in between (e.g., 3, 4).
                    new_A.extend(range(curr + 1, next_val))
                # If the numbers are decreasing (e.g., 5 to 1).
                else: # curr > next_val
                    # Insert the numbers in between (e.g., 4, 3, 2).
                    new_A.extend(range(curr - 1, next_val, -1))
                
                # After inserting, add the `next_val` itself.
                new_A.append(next_val)
                
                # The problem specifies modifying the first pair and restarting.
                # So, we append the rest of the original list and break this inner loop.
                new_A.extend(A[i+2:])
                break
            else:
                # If the difference is 1, no change is needed. Just append the next element.
                new_A.append(next_val)
                
        # Update A with the result of the pass.
        A = new_A
        
        # If the `for` loop completed without any changes, the process is done.
        if not made_change:
            break

    # Print the final list with elements separated by spaces.
    print(*A)

solve()