def max_satisfaction(N, cups):
    from collections import defaultdict
    
    # Dictionary to hold the maximum deliciousness for each flavor
    flavor_map = defaultdict(list)
    
    # Fill the flavor_map with deliciousness values
    for flavor, deliciousness in cups:
        flavor_map[flavor].append(deliciousness)
    
    # Sort the deliciousness values for each flavor in descending order
    for flavor in flavor_map:
        flavor_map[flavor].sort(reverse=True)
    
    # Prepare to find the maximum satisfaction
    max_satisfaction = 0
    max_deliciousness = []
    
    # Collect the top two deliciousness values from each flavor
    for deliciousness_list in flavor_map.values():
        if len(deliciousness_list) > 0:
            max_deliciousness.append(deliciousness_list[0])  # Top deliciousness
        if len(deliciousness_list) > 1:
            max_deliciousness.append(deliciousness_list[1])  # Second top deliciousness
    
    # Sort the collected deliciousness values in descending order
    max_deliciousness.sort(reverse=True)
    
    # Calculate maximum satisfaction
    if len(max_deliciousness) >= 2:
        # Case 1: Different flavors
        max_satisfaction = max(max_satisfaction, max_deliciousness[0] + max_deliciousness[1])
    
    # Now check for same flavor pairs
    for flavor, deliciousness_list in flavor_map.items():
        if len(deliciousness_list) > 1:
            # Same flavor, take the top two
            s = deliciousness_list[0]
            t = deliciousness_list[1]
            satisfaction = s + (t // 2)
            max_satisfaction = max(max_satisfaction, satisfaction)
    
    return max_satisfaction

import sys
input = sys.stdin.read
data = input().strip().split('
')
N = int(data[0])
cups = [tuple(map(int, line.split())) for line in data[1:N+1]]

print(max_satisfaction(N, cups))