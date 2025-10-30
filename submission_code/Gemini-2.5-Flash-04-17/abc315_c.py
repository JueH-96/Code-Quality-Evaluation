import sys
from collections import defaultdict

# Read input
N = int(sys.stdin.readline())
flavor_deliciousness = defaultdict(list)
for _ in range(N):
    F, S = map(int, sys.stdin.readline().split())
    flavor_deliciousness[F].append(S)

# Sort deliciousness lists for each flavor in descending order
# This makes finding max and second max easy (index 0 and 1)
for F in flavor_deliciousness:
    flavor_deliciousness[F].sort(reverse=True)

# 1. Calculate maximum same-flavor satisfaction
# Iterate through flavors that have at least two cups
max_same_flavor_sat = 0
for F, s_list in flavor_deliciousness.items():
    if len(s_list) >= 2:
        # The list is already sorted descending
        sat = s_list[0] + s_list[1] // 2
        max_same_flavor_sat = max(max_same_flavor_sat, sat)

# 2. Calculate maximum different-flavor satisfaction
# Find the maximum deliciousness for each distinct flavor
max_s_per_flavor_list = []
for F, s_list in flavor_deliciousness.items():
    # The max deliciousness for this flavor is s_list[0]
    max_s_per_flavor_list.append(s_list[0])

# If there are at least two distinct flavors, find the sum of the top two maximums
max_diff_flavor_sat = 0 # Default value if less than 2 distinct flavors
if len(max_s_per_flavor_list) >= 2:
    # Sort the list of maximum deliciousness values per flavor in descending order
    max_s_per_flavor_list.sort(reverse=True)
    max_diff_flavor_sat = max_s_per_flavor_list[0] + max_s_per_flavor_list[1]

# The overall maximum satisfaction is the maximum of the two cases
print(max(max_same_flavor_sat, max_diff_flavor_sat))