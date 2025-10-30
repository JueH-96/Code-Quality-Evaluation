import sys
import collections

def solve():
    N = int(sys.stdin.readline())
    
    all_cups_raw = []
    flavor_deliciousness = collections.defaultdict(list)
    
    # Read input and populate data structures
    for _ in range(N):
        F, S = map(int, sys.stdin.readline().split())
        all_cups_raw.append((F, S))
        flavor_deliciousness[F].append(S)
    
    max_overall_satisfaction = 0
    
    # --- Calculate maximum satisfaction for choosing two cups of the SAME flavor ---
    # For each flavor, we want the two highest deliciousness values (s1, s2) to calculate s1 + s2/2.
    for F in flavor_deliciousness:
        deliciousness_list = flavor_deliciousness[F]
        
        # Only consider flavors that have at least two cups
        if len(deliciousness_list) >= 2:
            # Sort the deliciousness values for this flavor in descending order
            # to easily get the top two.
            deliciousness_list.sort(reverse=True)
            s1 = deliciousness_list[0] # Highest deliciousness for this flavor
            s2 = deliciousness_list[1] # Second highest deliciousness for this flavor
            
            current_satisfaction = s1 + s2 // 2
            max_overall_satisfaction = max(max_overall_satisfaction, current_satisfaction)
            
    # --- Calculate maximum satisfaction for choosing two cups of DIFFERENT flavors ---
    # We want to find the two highest deliciousness values (s_max1, s_max2_candidate)
    # such that their flavors are different, to calculate s_max1 + s_max2_candidate.
    
    # Sort all cups by their deliciousness in descending order.
    # Each element in sorted_all_cups is a tuple (F, S).
    sorted_all_cups = sorted(all_cups_raw, key=lambda x: x[1], reverse=True)
    
    # The cup with the highest deliciousness overall is sorted_all_cups[0]
    f_max1, s_max1 = sorted_all_cups[0]
    
    # We need to find the highest deliciousness from a cup that has a different flavor than f_max1.
    s_max2_candidate = -1 # Initialize with a sentinel value indicating not found
    
    # Iterate through the sorted cups starting from the second element
    for i in range(1, N):
        current_f, current_s = sorted_all_cups[i]
        
        if current_f != f_max1:
            # Found the second desired cup (highest deliciousness with a different flavor)
            s_max2_candidate = current_s
            break # No need to look further
            
    # If a valid s_max2_candidate was found (i.e., not -1),
    # it means there are at least two distinct flavors available among the cups.
    if s_max2_candidate != -1:
        current_satisfaction = s_max1 + s_max2_candidate
        max_overall_satisfaction = max(max_overall_satisfaction, current_satisfaction)
        
    # Print the final maximum achievable satisfaction.
    # The problem constraints ensure S_i are integers, and S_i is even for S_i/2.
    # All calculations result in integers, so converting to int is not strictly needed
    # if `max_overall_satisfaction` is already an integer, but harmless.
    sys.stdout.write(str(int(max_overall_satisfaction)) + "
")

# Call the solve function to run the program
solve()