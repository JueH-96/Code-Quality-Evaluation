import sys

def solve():
    # Read N, the initial length of the sequence (not directly used but part of input spec)
    N = int(sys.stdin.readline())
    
    # Read the initial sequence A as a list of integers
    A = list(map(int, sys.stdin.readline().split()))

    # Main procedure loop: continues until the termination condition is met
    while True:
        # Flag to track if any insertion was performed in the current iteration
        inserted_this_iteration = False
        
        # Iterate through the current sequence to find the first pair needing insertion
        # The loop goes up to len(A) - 2 because we check A[i] and A[i+1]
        i = 0
        while i < len(A) - 1:
            current_val = A[i]
            next_val = A[i+1]

            # Check if the absolute difference between adjacent terms is not 1
            if abs(current_val - next_val) != 1:
                # A pair needing insertion is found (this is the first such pair)
                inserted_this_iteration = True
                
                # Determine the numbers to insert between current_val and next_val
                to_insert = []
                if current_val < next_val:
                    # If A_i < A_{i+1}, insert A_i+1, A_i+2, ..., A_{i+1}-1
                    # range(start, stop) generates numbers from 'start' up to 'stop-1'
                    for val in range(current_val + 1, next_val):
                        to_insert.append(val)
                else: # current_val > next_val
                    # If A_i > A_{i+1}, insert A_i-1, A_i-2, ..., A_{i+1}+1
                    # range(start, stop, step) with a negative step, 'stop' is exclusive
                    # e.g., range(4, 1, -1) generates 4, 3, 2
                    for val in range(current_val - 1, next_val, -1):
                        to_insert.append(val)
                
                # Reconstruct the sequence with the newly inserted elements
                # A[:i+1] gets elements from index 0 up to i (inclusive)
                # A[i+1:] gets elements from index i+1 to the end
                A = A[:i+1] + to_insert + A[i+1:]
                
                # According to the procedure, after finding the *first* such pair
                # and making an insertion, we "Return to step 1".
                # This means we break from the inner loop and the outer `while True`
                # will restart, causing a new scan from the beginning of the (now modified) sequence.
                break 
            
            i += 1
        
        # If the inner loop completed without finding any pair needing insertion
        # (i.e., 'inserted_this_iteration' remains False)
        if not inserted_this_iteration:
            # This means all adjacent differences are 1. The procedure terminates.
            break
    
    # Print the final sequence elements separated by spaces
    # map(str, A) converts all integers in list A to strings
    # " ".join(...) joins these strings with a space in between
    # Add a newline character at the end as is standard for output
    sys.stdout.write(" ".join(map(str, A)) + "
")

# Call the function to execute the program
solve()