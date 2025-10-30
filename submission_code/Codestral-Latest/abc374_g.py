import sys
from collections import defaultdict

def find_min_ng_list(N, product_names):
    # Create a dictionary to store the positions of each product name
    positions = defaultdict(list)
    for i, name in enumerate(product_names):
        positions[name].append(i)

    # Sort the product names by their first occurrence
    sorted_names = sorted(product_names, key=lambda x: positions[x][0])

    # Initialize the NG list
    ng_list = []
    current_string = ""

    # Iterate through the sorted product names
    for name in sorted_names:
        if name not in current_string:
            current_string += name
        else:
            ng_list.append(current_string)
            current_string = name

    # Add the last string to the NG list
    ng_list.append(current_string)

    return len(ng_list)

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
product_names = data[1:N+1]

# Find the minimum possible number of strings in the NG list
result = find_min_ng_list(N, product_names)

# Print the result
print(result)