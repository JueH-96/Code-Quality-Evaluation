# YOUR CODE HERE
import sys
from collections import Counter

def solve():
    """
    Reads 7 card values from standard input, determines if a full house 
    can be formed by choosing 5 of these cards, and prints "Yes" or "No" 
    to standard output.

    A full house requires selecting 5 cards such that there are exactly 
    three cards of one value (x) and exactly two cards of another value (y), 
    where x and y are different.

    This is possible if and only if the original set of 7 cards contains 
    at least 3 cards of some value x AND at least 2 cards of some 
    *different* value y.
    """
    # Read the list of 7 integers from standard input
    try:
        a = list(map(int, sys.stdin.readline().split()))
        # Assuming the input format is always correct as per problem statement
        # No need for explicit check of len(a) == 7 in competitive programming context
    except ValueError:
        # Handle potential errors if input is not numbers
        # print("Invalid input format.", file=sys.stderr)
        return # Exit if input is malformed

    # Count the frequency of each card value using collections.Counter
    counts_map = Counter(a)
    
    # We need to determine if there exist two distinct card values, x and y,
    # such that the count of x is at least 3 and the count of y is at least 2.

    # Let's count how many distinct card values have a frequency of at least 3
    num_distinct_values_ge3 = 0 
    # Let's count how many distinct card values have a frequency of at least 2
    num_distinct_values_ge2 = 0 
    
    # Iterate through the frequencies (counts) of the distinct card values
    for count in counts_map.values():
        if count >= 3:
            num_distinct_values_ge3 += 1
        # A value with count >= 3 also has count >= 2.
        # We count the total number of distinct values with count >= 2.
        if count >= 2:
            num_distinct_values_ge2 += 1
            
    # Initialize the possibility flag to False
    possible = False
    
    # Check if the conditions for forming a full house can be met.
    # Condition 1: We must have at least one card value (x) with a count of 3 or more.
    #              (num_distinct_values_ge3 >= 1)
    # Condition 2: We must have at least one card value (y) with a count of 2 or more.
    #              (num_distinct_values_ge2 >= 1)
    # Condition 3: The value x and value y must be different.
    
    # If conditions 1 and 2 are met, when can condition 3 fail?
    # Condition 3 fails ONLY if the set of values satisfying count >= 3 
    # is identical to the set of values satisfying count >= 2, AND 
    # both sets contain exactly one element.
    # This means there is exactly one value k with count >= 3, and 
    # this value k is also the only value with count >= 2.
    # This scenario happens if and only if:
    # num_distinct_values_ge3 == 1 AND num_distinct_values_ge2 == 1.
    
    # Therefore, a full house is possible if Condition 1 and Condition 2 are met,
    # AND the failure case (num_ge3 == 1 and num_ge2 == 1) does NOT occur.
    
    if num_distinct_values_ge3 >= 1 and num_distinct_values_ge2 >= 1:
        # Check if we are NOT in the specific failure case
        if not (num_distinct_values_ge3 == 1 and num_distinct_values_ge2 == 1):
            possible = True

    # Print the final result to standard output
    if possible:
        print("Yes")
    else:
        print("No")

# Execute the solve function when the script is run
if __name__ == '__main__':
    solve()