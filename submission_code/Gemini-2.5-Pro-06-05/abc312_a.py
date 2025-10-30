# YOUR CODE HERE
def solve():
    """
    Solves the problem by reading a 3-letter string from standard input
    and checking if it belongs to a predefined set of strings.
    """
    # Read the input string from standard input
    s = input()

    # Define the set of target strings for efficient lookup
    target_strings = {"ACE", "BDF", "CEG", "DFA", "EGB", "FAC", "GBD"}

    # Check if the input string is in the set of target strings
    if s in target_strings:
        # If it is, print "Yes"
        print("Yes")
    else:
        # Otherwise, print "No"
        print("No")

# Execute the solution function
solve()