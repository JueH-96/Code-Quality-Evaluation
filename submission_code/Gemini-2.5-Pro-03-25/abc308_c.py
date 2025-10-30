import sys
import functools

# Define the comparison function for sorting
# It compares two items based on success rate descending, then index ascending.
def compare(item1, item2):
    """
    Comparison function for two items representing people.
    item format: (index, A_heads, B_tails)
    Compares based on success rate A / (A + B) descending.
    Ties are broken by index ascending.
    Uses cross-multiplication to avoid floating point issues and maintain precision.
    
    Returns:
        -1 if item1 should come before item2 in the sorted order.
         1 if item1 should come after item2 in the sorted order.
         0 is not returned explicitly as indices i and j are always distinct for different items.
           However, the logic implicitly covers the standard comparison function contract.
    """
    # Unpack items: index, heads count (A), tails count (B)
    i, Ai, Bi = item1
    j, Aj, Bj = item2
    
    # Calculate the terms needed for cross-multiplication comparison.
    # We want to compare the success rates: Ai / (Ai + Bi) versus Aj / (Aj + Bj).
    # This comparison is equivalent to comparing Ai * (Aj + Bj) versus Aj * (Ai + Bi),
    # given that the denominators (Ai + Bi) and (Aj + Bj) are positive.
    # The problem constraints guarantee Ai + Bi >= 1 and Aj + Bj >= 1.
    
    # Python's integers have arbitrary precision, so they can handle large numbers
    # that might arise from multiplication (up to 2 * 10^18).
    val1 = Ai * (Aj + Bj) 
    val2 = Aj * (Ai + Bi)
    
    # Primary sorting criterion: success rate in descending order.
    # If item1 has a higher success rate, val1 > val2.
    # In standard sorting (ascending), the item that comes first is considered "smaller".
    # So if val1 > val2, item1 should come first, meaning item1 is "smaller". Return -1.
    if val1 > val2:
        return -1 
    # If item1 has a lower success rate, val1 < val2.
    # item2 should come first, meaning item1 is "larger". Return 1.
    elif val1 < val2:
        return 1
    # If success rates are equal (val1 == val2).
    else:
        # Secondary sorting criterion: person index in ascending order.
        # If item1 has a smaller index (i < j), it should come first. Return -1.
        if i < j:
            return -1
        # Otherwise (i > j, since indices are unique for distinct items), 
        # item2 has a smaller index and should come first. item1 is "larger". Return 1.
        else: 
            return 1

# Main execution logic encapsulated in a function
def solve():
    # Read the number of people N from standard input
    N = int(sys.stdin.readline())
    
    # Initialize a list to store data for each person
    people = []
    # Loop N times, reading data for each person indexed from 1 to N
    for k in range(1, N + 1): 
        # Read the number of heads (A) and tails (B) for person k
        A, B = map(int, sys.stdin.readline().split())
        # Store the person's data as a tuple: (index, A_heads, B_tails)
        people.append((k, A, B))

    # Sort the list `people` using the custom comparison function `compare`.
    # `functools.cmp_to_key` converts the old-style comparison function (`compare`) 
    # into a key function that can be used with `sort`'s `key` argument.
    people.sort(key=functools.cmp_to_key(compare))

    # After sorting, the `people` list contains tuples ordered according to the criteria.
    # Extract the sorted indices (the first element `p[0]` of each tuple `p`).
    sorted_indices = [p[0] for p in people]
    
    # Print the sorted indices to standard output.
    # The indices should be separated by spaces on a single line.
    # The `*` operator unpacks the `sorted_indices` list into individual arguments for `print`.
    print(*sorted_indices)

# Execute the main logic by calling the solve function.
solve()