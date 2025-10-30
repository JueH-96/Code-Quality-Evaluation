import sys

def solve():
    # Read the input sequence A from standard input.
    # The input is space-separated integers on a single line.
    A_str = sys.stdin.readline().split()
    # Convert the string elements to integers.
    A = [int(x) for x in A_str]

    # The target sorted sequence for comparison.
    # Since A is a permutation of (1,2,3,4,5), the sorted order is always this.
    target_sorted_A = [1, 2, 3, 4, 5]

    # A flag to keep track if we found a valid swap that sorts the array.
    found_solution = False

    # Iterate through all possible pairs of adjacent elements to swap.
    # For an array of length 5, there are 4 possible adjacent swap positions:
    # (A[0], A[1]), (A[1], A[2]), (A[2], A[3]), (A[3], A[4])
    # The loop iterates with 'i' as the first index of the pair, from 0 to len(A) - 2.
    for i in range(len(A) - 1):
        # Create a temporary copy of the list A.
        # This is crucial because we need to perform swaps on a fresh copy
        # for each iteration, without altering the original list A for subsequent checks.
        temp_A = list(A)
        
        # Perform the swap of elements at index i and i+1 in the temporary list.
        temp_A[i], temp_A[i+1] = temp_A[i+1], temp_A[i]
        
        # After performing the swap, check if the modified temporary list is now sorted
        # (i.e., it matches the target_sorted_A).
        if temp_A == target_sorted_A:
            found_solution = True
            # If a solution is found, we can stop checking further swaps as
            # the problem asks if it *can* be sorted by *exactly one* operation.
            break 
    
    # Print "Yes" if a valid swap was found, otherwise print "No".
    # Use sys.stdout.write for competitive programming style, and ensure a newline character.
    if found_solution:
        sys.stdout.write("Yes
")
    else:
        sys.stdout.write("No
")

# Call the solve function to execute the program logic.
solve()