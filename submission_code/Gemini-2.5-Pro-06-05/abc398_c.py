# YOUR CODE HERE
import sys
from collections import Counter

def main():
    """
    This function reads the input, solves the problem, and prints the output.
    The overall time complexity is O(N) and space complexity is O(U), where N
    is the number of people and U is the number of unique integers.
    """
    # Read the number of people, N. Using sys.stdin.readline for efficiency.
    try:
        n = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # This handles cases of empty or malformed input for N.
        # In a typical contest setting with guaranteed valid input, this might
        # be omitted, but it adds robustness.
        return
        
    # Read the list of integers A_i associated with each person.
    a = list(map(int, sys.stdin.readline().split()))

    # Use collections.Counter to efficiently count the frequency of each integer.
    # This is a hash map based approach, with an average time complexity of O(N).
    counts = Counter(a)

    # Initialize variables to track the best candidate.
    # A person is a candidate if their associated integer is unique.
    # We want the candidate with the largest integer value.
    # `max_unique_val` stores the largest unique integer found so far.
    # `best_label` stores the 1-based label of the person with that integer.
    # We initialize `best_label` to -1, which is the required output
    # if no person with a unique integer is found.
    max_unique_val = -1
    best_label = -1

    # Iterate through the list of people. The `enumerate` function provides both
    # the 0-based index `i` and the value `val` for each person. This is an O(N) operation.
    for i, val in enumerate(a):
        # Check if the integer `val` is unique by looking up its count.
        # This lookup is an O(1) average time operation in a hash map.
        if counts[val] == 1:
            # If the integer is unique, this person is a candidate.
            # We then check if this candidate's integer is greater than the
            # largest unique integer we have found so far.
            if val > max_unique_val:
                # If it is, we have a new best candidate.
                # Update the maximum value and the corresponding label.
                max_unique_val = val
                best_label = i + 1  # The labels are 1-based, index is 0-based.

    # After iterating through all people, print the label of the best candidate.
    # If no candidates were found, `best_label` will have its initial value of -1.
    print(best_label)

# The standard entry point for a Python script.
# The code inside this block will run only when the script is executed directly.
if __name__ == "__main__":
    main()