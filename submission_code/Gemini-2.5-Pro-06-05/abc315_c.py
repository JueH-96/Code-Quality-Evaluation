import sys
from collections import defaultdict

def solve():
    """
    Reads ice cream data, calculates the maximum possible satisfaction,
    and prints the result.
    """
    # Use fast I/O, a common practice in competitive programming
    input = sys.stdin.readline

    try:
        N = int(input())
    except (IOError, ValueError):
        # Handle empty input for local testing
        return

    # Use a defaultdict to group deliciousness scores by flavor.
    # Keys are flavors, values are lists of deliciousness scores.
    flavors = defaultdict(list)
    for _ in range(N):
        f, s = map(int, input().split())
        flavors[f].append(s)

    # This will hold the maximum satisfaction found. Initialize to 0.
    max_satisfaction = 0

    # This list will store the highest deliciousness score for each distinct flavor.
    # It will be used to find the best pair of different-flavored cups.
    best_per_flavor = []

    # Process each flavor and its scores.
    for f in flavors:
        # Sort scores in descending order to easily find the best ones.
        flavors[f].sort(reverse=True)
        
        # The best score for this flavor is the first element.
        best_per_flavor.append(flavors[f][0])
        
        # --- Case 1: Two cups of the SAME flavor ---
        # This is possible only if there are at least two cups of this flavor.
        if len(flavors[f]) >= 2:
            s_large = flavors[f][0]
            s_small = flavors[f][1]
            
            # Satisfaction is s_large + s_small / 2.
            # Integer division // is used since S_i is even.
            current_satisfaction = s_large + s_small // 2
            
            # Update the overall maximum if this pair is better.
            max_satisfaction = max(max_satisfaction, current_satisfaction)

    # --- Case 2: Two cups of DIFFERENT flavors ---
    # This is possible only if there are at least two distinct flavors.
    if len(best_per_flavor) >= 2:
        # Sort the list of bests to find the top two overall.
        best_per_flavor.sort(reverse=True)
        
        # The top two in this list are the best scores from two different flavors.
        s1 = best_per_flavor[0]
        s2 = best_per_flavor[1]
        
        # Satisfaction is s1 + s2.
        current_satisfaction = s1 + s2
        
        # Update the overall maximum if this pair is better.
        max_satisfaction = max(max_satisfaction, current_satisfaction)

    # Print the final answer.
    print(max_satisfaction)

solve()