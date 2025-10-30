import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()
index = 0

# Read Q, the number of queries
Q = int(data[index])
index += 1

# Initialize a dictionary to keep track of the count of each number
count_dict = {}

# Process each query
for _ in range(Q):
    query_type = int(data[index])
    index += 1
    
    if query_type == 1:
        # Add a ball with number x
        x = int(data[index])
        index += 1
        count_dict[x] = count_dict.get(x, 0) + 1
    elif query_type == 2:
        # Remove a ball with number x
        x = int(data[index])
        index += 1
        count_dict[x] -= 1
        if count_dict[x] == 0:
            del count_dict[x]
    elif query_type == 3:
        # Print the number of distinct integers in the bag
        print(len(count_dict))