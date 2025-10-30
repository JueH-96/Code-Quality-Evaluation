import sys
from itertools import combinations_with_replacement

def solve():
    """
    This function reads an integer N and prints the N-th smallest number
    that is a sum of exactly three repunits.
    """
    try:
        # Read the integer N from standard input.
        N = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # This path is unlikely given the problem constraints.
        return

    # Generate the first 12 repunits. As determined by combinatorial analysis,
    # this is sufficient to find up to the 333rd smallest sum.
    repunits = []
    current_repunit = 0
    for _ in range(12):
        current_repunit = current_repunit * 10 + 1
        repunits.append(current_repunit)
    
    # Use itertools.combinations_with_replacement to get all unique sets
    # of three repunits.
    three_repunits_combos = combinations_with_replacement(repunits, 3)
    
    # Calculate the sum for each combination and store it in a list.
    sums = [sum(combo) for combo in three_repunits_combos]
    
    # Sort the list of sums to find the N-th smallest.
    sums.sort()
    
    # Print the N-th smallest sum. N is 1-indexed, so we access index N-1.
    print(sums[N - 1])

solve()