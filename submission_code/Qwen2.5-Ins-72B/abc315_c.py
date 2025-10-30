# YOUR CODE HERE
import sys
from collections import defaultdict

def max_satisfaction(N, cups):
    flavor_dict = defaultdict(list)
    
    for flavor, deliciousness in cups:
        flavor_dict[flavor].append(deliciousness)
    
    for flavor in flavor_dict:
        flavor_dict[flavor].sort(reverse=True)
    
    max_diff_flavor = 0
    max_same_flavor = 0
    
    # Calculate max satisfaction for different flavors
    max_deliciousness = [0, 0]
    for flavor in flavor_dict:
        max_deliciousness[0] = max(max_deliciousness[0], flavor_dict[flavor][0])
        if len(flavor_dict[flavor]) > 1:
            max_deliciousness[1] = max(max_deliciousness[1], flavor_dict[flavor][1])
    max_diff_flavor = sum(max_deliciousness)
    
    # Calculate max satisfaction for same flavors
    for flavor in flavor_dict:
        if len(flavor_dict[flavor]) > 1:
            max_same_flavor = max(max_same_flavor, flavor_dict[flavor][0] + flavor_dict[flavor][1] // 2)
    
    return max(max_diff_flavor, max_same_flavor)

# Read input
N = int(input())
cups = [tuple(map(int, input().split())) for _ in range(N)]

# Compute and print the result
print(max_satisfaction(N, cups))