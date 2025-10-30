from collections import defaultdict

# Read input
N = int(input())
A = list(map(int, input().split()))

# Dictionary to store indices of each value
index_dict = defaultdict(list)

# Populate the dictionary with indices
for i in range(N):
    index_dict[A[i]].append(i)

# Initialize the count of valid triples
count = 0

# Iterate through the dictionary
for key in index_dict:
    indices = index_dict[key]
    m = len(indices)
    if m >= 2:
        # Count the number of ways to choose two different indices from the list
        count += m * (m - 1) // 2

# Print the result
print(count)