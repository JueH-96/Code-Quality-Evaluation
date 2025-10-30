# YOUR CODE HERE
import sys

def solve():
    """
    Reads input, processes treatments, and calculates the final number of teeth.
    """
    # Read N (number of holes/initial teeth) and Q (number of treatments)
    # from the first line of standard input.
    n, q = map(int, sys.stdin.readline().split())

    # Read the sequence of treatments T_1, T_2, ..., T_Q
    # from the second line of standard input.
    # Constraints guarantee Q >= 1, so this line will always have input values.
    treatments = list(map(int, sys.stdin.readline().split()))

    # Use a set to keep track of the indices (1-based) of the holes that are currently empty.
    # Initially, all N holes have teeth, so the set of empty holes is empty.
    # A hole T_i will end up in this set if it has been treated an odd number of times.
    empty_holes = set()

    # Process each treatment one by one
    for t in treatments:
        # Check if the hole 't' is currently in the set of empty holes.
        if t in empty_holes:
            # If hole 't' is currently empty (i.e., t is in the set),
            # the treatment causes a tooth to grow in it.
            # This means the hole is no longer empty.
            # We remove 't' from the set of empty holes.
            empty_holes.remove(t)
        else:
            # If hole 't' currently has a tooth (i.e., t is not in the set),
            # the treatment removes the tooth.
            # This means the hole becomes empty.
            # We add 't' to the set of empty holes.
            empty_holes.add(t)

    # After processing all Q treatments, the set `empty_holes` contains the indices
    # of all holes that are empty.
    # The total number of teeth remaining is the initial number of teeth (N)
    # minus the number of holes that are currently empty.
    # The number of empty holes is simply the size (length) of the `empty_holes` set.
    final_teeth_count = n - len(empty_holes)

    # Print the final count of teeth to standard output.
    print(final_teeth_count)

# Call the main function to execute the logic when the script runs.
solve()