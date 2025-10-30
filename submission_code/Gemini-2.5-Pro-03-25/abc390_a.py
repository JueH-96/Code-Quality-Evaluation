# YOUR CODE HERE
import sys

# Wrap the main logic in a function
def solve():
    """
    Reads a sequence of 5 integers (a permutation of 1-5),
    determines if it can be sorted into ascending order (1, 2, 3, 4, 5)
    by swapping exactly one pair of adjacent elements, and prints "Yes" or "No".
    """
    # Read the input sequence A from standard input
    # Input format: A_1 A_2 A_3 A_4 A_5
    # Example: 1 2 4 3 5
    # The input line is read, split by spaces, and each part is converted to an integer.
    a = list(map(int, sys.stdin.readline().split()))
    
    # Define the target sorted sequence
    target = [1, 2, 3, 4, 5]
    
    # Flag to indicate if a successful adjacent swap was found
    # that results in the sorted sequence. Initialize to False.
    found_swap = False
    
    # Iterate through all possible adjacent swap positions.
    # For a list of length 5 (indices 0, 1, 2, 3, 4),
    # the adjacent pairs are at indices (0, 1), (1, 2), (2, 3), (3, 4).
    # The loop index 'i' will represent the first index of the pair to swap.
    # It needs to go from 0 up to len(a) - 2, which is 3 for length 5.
    # range(len(a) - 1) correctly generates indices 0, 1, 2, 3.
    for i in range(len(a) - 1): 
        # Create a copy of the original list 'a' for each potential swap.
        # This is crucial because we need to test each swap independently
        # starting from the original state of the list 'a'.
        temp_a = a.copy() 
        
        # Perform the adjacent swap at indices i and i+1 on the temporary copy.
        # Python's tuple assignment makes swapping concise.
        temp_a[i], temp_a[i+1] = temp_a[i+1], temp_a[i]
        
        # Check if the resulting list 'temp_a' after the swap
        # is identical to the target sorted list.
        if temp_a == target:
            # If they are equal, it means we found an adjacent swap
            # that sorts the list. The problem asks if exactly one such
            # operation can sort the list. Since 'a' is a permutation,
            # it can be shown that at most one adjacent swap can result
            # in the sorted list. Therefore, finding one such swap is sufficient.
            found_swap = True
            # Once we find the required swap, there's no need to check
            # the remaining possible swaps. We can exit the loop early.
            break 
            
    # After the loop finishes (either by checking all swaps or breaking early),
    # we check the value of the 'found_swap' flag.
    if found_swap:
        # If the flag is True, it means a single adjacent swap was found
        # that sorts the list. Print "Yes".
        print("Yes")
    else:
        # If the flag is False, it means no single adjacent swap resulted
        # in the sorted list. This covers cases where the list was already
        # sorted (requires 0 swaps) or needs more than one swap. Print "No".
        print("No")

# Execute the solve function to run the program
solve()