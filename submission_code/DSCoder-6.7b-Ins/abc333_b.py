# YOUR CODE HERE

import sys

# Read the inputs
S_1, S_2 = sys.stdin.readline().strip().split()
T_1, T_2 = sys.stdin.readline().strip().split()

# Define the distances between the points
distances = {
    'AB': 1, 'AC': 1, 'AD': 2, 'AE': 2,
    'BA': 1, 'BC': 2, 'BD': 2, 'BE': 3,
    'CA': 1, 'CB': 2, 'CD': 1, 'CE': 2,
    'DA': 2, 'DB': 2, 'DC': 1, 'DE': 1,
    'EA': 2, 'EB': 3, 'EC': 2, 'ED': 1
}

# Check if the lengths of the line segments are equal
if distances[S_1 + S_2] == distances[T_1 + T_2]:
    print('Yes')
else:
    print('No')