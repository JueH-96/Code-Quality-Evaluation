import sys
from collections import defaultdict

# Read all input from stdin
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1

# Group the deliciousness values by flavor
flavor_dict = defaultdict(list)
for _ in range(N):
    F = int(data[index])
    index += 1  # Move to S
    S = int(data[index])  # Read S
    index += 1  # Move to next pair
    flavor_dict[F].append(S)

# Initialize maximum satisfaction for same flavor and list for max S per flavor
max_same_sat = -1  # Sentinel value less than minimum possible satisfaction
max_S_per_flavor_list = []

# Iterate through each flavor to compute max S and max same flavor satisfaction if applicable
for flavor in flavor_dict:
    lst = flavor_dict[flavor]
    max_S = max(lst)  # Maximum deliciousness for this flavor
    max_S_per_flavor_list.append(max_S)
    
    if len(lst) >= 2:
        # Sort the list in descending order to find the two largest values
        sorted_lst = sorted(lst, reverse=True)
        second_max_S = sorted_lst[1]
        sat = max_S + (second_max_S // 2)  # Satisfaction for same flavor
        max_same_sat = max(max_same_sat, sat)

# Compute maximum satisfaction for different flavors
num_flavors = len(max_S_per_flavor_list)
if num_flavors >= 2:
    # Sort the max S values in descending order and sum the two largest
    sorted_max_S = sorted(max_S_per_flavor_list, reverse=True)
    max_diff_sum = sorted_max_S[0] + sorted_max_S[1]
else:
    max_diff_sum = -1  # Sentinel value if not possible

# The answer is the maximum of the two satisfaction values
ans = max(max_same_sat, max_diff_sum)

# Output the answer
print(ans)