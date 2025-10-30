# YOUR CODE HERE
import sys

def solve():
    # Read the lengths of the sequences N and M from the first line of input.
    # sys.stdin.readline() reads a line from standard input.
    # .split() splits the line into parts based on whitespace.
    # map(int, ...) converts each part into an integer.
    n, m = map(int, sys.stdin.readline().split())
    
    # Read the sequence A from the second line of input.
    # Convert the space-separated numbers into a list of integers.
    a = list(map(int, sys.stdin.readline().split()))
    
    # Read the sequence B from the third line of input.
    # Convert the space-separated numbers into a list of integers.
    b = list(map(int, sys.stdin.readline().split()))

    # Combine the two lists A and B into a single list C.
    # The problem statement guarantees that all elements in A and B are initially pairwise distinct.
    c = a + b
    
    # Sort the combined list C in ascending order. This places all elements
    # from both original sequences into their correct sorted order.
    c.sort()

    # Create a set from list A. Sets provide efficient membership testing (checking if an element is in the set).
    # On average, checking `x in set_a` takes O(1) time. This is much faster than checking `x in a` (list), which takes O(N) time.
    set_a = set(a)

    # Initialize a boolean flag to keep track of whether we have found the condition specified in the problem.
    # The condition is: are there two consecutive elements in the sorted list C
    # such that both of these elements were originally present in list A?
    found_consecutive_pair_from_a = False
    
    # Iterate through the sorted list C. We need to examine adjacent pairs of elements (C[i], C[i+1]).
    # The loop should go from the first element up to the second-to-last element.
    # The total number of elements in C is n + m. Indices range from 0 to (n + m - 1).
    # To check pairs (C[i], C[i+1]), the index `i` needs to go from 0 up to (n + m - 2).
    # The `range(n + m - 1)` generates integers from 0 up to (n + m - 1) - 1 = n + m - 2, which is exactly what we need.
    list_length = n + m
    for i in range(list_length - 1):
        # Get the current element C[i] from the sorted list.
        element1 = c[i]
        # Get the next element C[i+1] from the sorted list.
        element2 = c[i+1]
        
        # Check if both element1 and element2 were originally part of sequence A.
        # We use the pre-computed set `set_a` for fast lookups.
        if element1 in set_a and element2 in set_a:
            # If both elements are from A and they are consecutive in the sorted combined list C,
            # then we have found an instance that satisfies the problem's condition.
            # Set the flag to True.
            found_consecutive_pair_from_a = True
            # Since the problem only asks if *such a pair exists*, we don't need to continue checking.
            # We can break out of the loop as soon as the first such pair is found.
            break

    # After the loop has finished (either by completing all iterations or breaking early),
    # we check the value of the flag to determine the final output.
    if found_consecutive_pair_from_a:
        # If the flag is True, it means we found at least one pair of consecutive elements in C
        # where both elements came from A.
        print("Yes")
    else:
        # If the flag is False, it means that after checking all consecutive pairs in C,
        # none of them consisted of two elements both originating from A.
        print("No")

# Call the solve function to execute the program logic.
solve()