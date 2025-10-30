# YOUR CODE HERE
import sys
from collections import Counter

# Define the main logic within a function for clarity and potential reuse
def solve():
    """
    Reads the input, finds the person with the largest unique integer,
    and prints their label or -1 if no such person exists.
    """
    # Read the number of people from standard input
    n = int(sys.stdin.readline())
    # Read the list of integers associated with each person from standard input
    a = list(map(int, sys.stdin.readline().split()))

    # Use collections.Counter to efficiently count the occurrences of each integer in list 'a'.
    # The keys of 'counts' will be the integers from 'a', and the values will be their frequencies.
    counts = Counter(a)

    # Initialize variables to keep track of the best candidate found so far.
    # 'max_unique_value' stores the largest integer found so far that appears exactly once.
    # Initialize it to -1, assuming all A_i are positive as per constraints (1 <= A_i).
    max_unique_value = -1
    # 'result_label' stores the 1-based index (label) of the person associated with 'max_unique_value'.
    # Initialize it to -1, which will be the output if no person satisfies the condition.
    result_label = -1

    # Iterate through the list of people. 'i' represents the 0-based index (from 0 to n-1).
    for i in range(n):
        # Get the integer associated with the current person (at index 'i').
        current_value = a[i]
        # Calculate the 1-based label for the current person.
        current_label = i + 1

        # Check if the integer 'current_value' appears exactly once in the list 'a'.
        # We look up its count in the pre-computed 'counts' dictionary.
        if counts[current_value] == 1:
            # If the current person's integer is unique:
            # Check if this unique integer is greater than the maximum unique integer found so far.
            if current_value > max_unique_value:
                # If it is, update 'max_unique_value' to this new larger value.
                max_unique_value = current_value
                # Update 'result_label' to the label of the current person.
                result_label = current_label

    # After iterating through all people, print the final 'result_label'.
    # If no unique integers were found, 'result_label' remains -1.
    # Otherwise, it holds the label of the person with the largest unique integer.
    print(result_label)

# Execute the solve function to run the program logic
solve()