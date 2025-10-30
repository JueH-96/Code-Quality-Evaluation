# YOUR CODE HERE
def solve():
    """
    Reads a sequence of 5 integers, a permutation of (1,2,3,4,5),
    and determines if it can be sorted in ascending order by performing
    exactly one swap of adjacent elements.
    """
    # Read the sequence of 5 integers from stdin.
    try:
        A = list(map(int, input().split()))
    except (IOError, ValueError):
        # Handle potential errors with input, though constraints imply valid input.
        return

    # Define the target sorted sequence.
    sorted_sequence = [1, 2, 3, 4, 5]

    # Iterate through all 4 possible adjacent swaps.
    for i in range(4):
        # Create a copy of the original list for manipulation.
        temp_A = A[:]
        
        # Perform the adjacent swap.
        temp_A[i], temp_A[i+1] = temp_A[i+1], temp_A[i]
        
        # Check if the list is now sorted.
        if temp_A == sorted_sequence:
            # If a single adjacent swap sorts the list, the condition is met.
            print("Yes")
            return

    # If the loop completes, no single adjacent swap could sort the list.
    print("No")

# Run the solution.
solve()