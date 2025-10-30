# YOUR CODE HERE
import sys

# Function to solve the problem
def solve():
    # Read the size of the permutation
    N = int(sys.stdin.readline())
    
    # Read the permutation A. Store it in a list using 0-based indexing.
    # The values in A are 1-based (from 1 to N).
    A = list(map(int, sys.stdin.readline().split()))

    # Create a position array 'pos'. pos[v] will store the 0-based index
    # where the value 'v' is currently located in the list A.
    # The size is N+1 because values are from 1 to N (index 0 of pos is unused).
    pos = [0] * (N + 1)
    
    # Initialize the position array by iterating through the initial permutation A.
    for i in range(N):
        pos[A[i]] = i

    # List to store the swaps performed. Each element will be a tuple (i, j)
    # where i and j are 1-based indices of the swapped elements.
    # The problem requires 1 <= i < j <= N.
    swaps = []

    # Iterate through each index i from 0 to N-1.
    # The goal is to place the value i+1 at index i.
    for i in range(N):
        # If the element currently at index i is not the value it should be (which is i+1),
        # we need to perform a swap to put the correct value here.
        if A[i] != i + 1:
            # The value that belongs at index i is i+1.
            value_to_place = i + 1

            # Find the current position 'p' (0-based index) of the value_to_place.
            # We use the 'pos' array for an O(1) lookup.
            current_pos_of_value = pos[value_to_place]

            # The value currently at index i is A[i]. We need to know this value
            # because its position will change after the swap, and we must update 'pos'.
            value_at_current_i = A[i]

            # Perform the swap: swap the element at index i with the element
            # at current_pos_of_value.
            A[i], A[current_pos_of_value] = A[current_pos_of_value], A[i]

            # Update the 'pos' array to reflect the new positions of the two swapped values.
            # The value_to_place (which is i+1) moved from current_pos_of_value to index i.
            pos[value_to_place] = i
            # The value_at_current_i (the value previously at index i) moved from index i
            # to current_pos_of_value.
            pos[value_at_current_i] = current_pos_of_value

            # Record this swap. The problem requires 1-based indexing for the output.
            # The indices involved in the swap are i (0-based) and current_pos_of_value (0-based).
            # Since A[i] != i+1, current_pos_of_value != i.
            # Also, since values 1..i are already fixed in positions 0..i-1 (by previous iterations),
            # the value i+1 cannot be in indices < i. Thus, current_pos_of_value must be >= i.
            # Since current_pos_of_value != i, it must be current_pos_of_value > i.
            # So, the 1-based indices are i+1 and current_pos_of_value+1, where i+1 < current_pos_of_value+1.
            # We add the pair (i+1, current_pos_of_value+1) to the swaps list.
            swaps.append((i + 1, current_pos_of_value + 1))

    # After the loop finishes, the array A is sorted as (1, 2, ..., N).

    # Print the total number of operations (swaps).
    print(len(swaps))
    
    # Print each swap operation (pair of 1-based indices).
    for swap in swaps:
        print(swap[0], swap[1])

# Call the solve function to run the program
solve()