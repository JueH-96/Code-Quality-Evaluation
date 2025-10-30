# YOUR CODE HERE
import sys
from collections import defaultdict

def solve():
    # Read the number of cups
    N = int(sys.stdin.readline())
    
    # Use a defaultdict to store lists of deliciousness values for each flavor
    flavor_map = defaultdict(list)
    
    # Read N lines of input, each containing flavor F and deliciousness S
    # Populate the flavor_map: keys are flavors, values are lists of deliciousness scores for that flavor
    for _ in range(N):
        # Read F and S, splitting the line and converting to integers
        line = sys.stdin.readline().split()
        F = int(line[0])
        S = int(line[1])
        # Append the deliciousness score to the list associated with flavor F
        flavor_map[F].append(S)

    # Initialize the maximum satisfaction found so far for the case where
    # two cups of the SAME flavor are chosen. Start with 0.
    max_satisfaction_same_flavor = 0
    
    # List to store the maximum deliciousness found for each distinct flavor
    # This will be used later to calculate the maximum satisfaction for different flavors.
    max_S_per_flavor_list = []

    # Iterate through each distinct flavor found in the input
    for F in flavor_map:
        # Get the list of deliciousness values for the current flavor F
        deliciousness_list = flavor_map[F]
        
        # Check if the list is empty. This case should not happen based on problem constraints (N>=2)
        # and how defaultdict works, but it's safe practice.
        if not deliciousness_list:
            continue

        # Sort the deliciousness values for the current flavor in descending order.
        # This makes it easy to find the highest and second highest deliciousness values.
        # The complexity is O(N_F log N_F) where N_F is the number of cups of flavor F.
        deliciousness_list.sort(reverse=True)
        
        # The highest deliciousness for flavor F is the first element after sorting.
        # Add this maximum deliciousness to our list that tracks the maximum for each flavor.
        max_S_per_flavor_list.append(deliciousness_list[0])

        # Check if there are at least two cups of the current flavor F.
        # If so, we can calculate the satisfaction for choosing two cups of this flavor.
        if len(deliciousness_list) >= 2:
            # s is the highest deliciousness value for flavor F.
            s = deliciousness_list[0]
            # t is the second highest deliciousness value for flavor F.
            t = deliciousness_list[1]
            
            # Calculate the satisfaction according to the rule for same flavors: s + t/2.
            # Since all S_i are given as even, t is even. Integer division // is appropriate.
            current_satisfaction = s + t // 2
            
            # Update the overall maximum satisfaction found so far for the same flavor case
            # if the current flavor provides a higher satisfaction.
            max_satisfaction_same_flavor = max(max_satisfaction_same_flavor, current_satisfaction)

    # Initialize the maximum satisfaction found so far for the case where
    # two cups of DIFFERENT flavors are chosen. Start with 0.
    max_satisfaction_diff_flavor = 0
    
    # To calculate the maximum satisfaction for choosing cups of different flavors,
    # we need at least two distinct flavors to have been present in the input.
    if len(max_S_per_flavor_list) >= 2:
        # Sort the list containing the maximum deliciousness value for each flavor in descending order.
        # The complexity is O(k log k) where k is the number of distinct flavors.
        max_S_per_flavor_list.sort(reverse=True)
        
        # The maximum satisfaction for different flavors is achieved by picking the two cups
        # that correspond to the two highest maximum deliciousness values across all distinct flavors.
        # The satisfaction value is simply the sum of these two highest maximum deliciousness values.
        max_satisfaction_diff_flavor = max_S_per_flavor_list[0] + max_S_per_flavor_list[1]

    # The final answer is the maximum of the satisfactions calculated for the two cases:
    # choosing two cups of the same flavor vs. choosing two cups of different flavors.
    print(max(max_satisfaction_same_flavor, max_satisfaction_diff_flavor))

# Execute the solve function to run the program logic.
solve()