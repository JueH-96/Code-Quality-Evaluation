import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()

# The first element is N
N = int(data[0])

# The next N elements are the array A
A = list(map(int, data[1: N + 1]))

# Dictionary to keep track of the last position of each number
last_pos = {}

# List to store the result B
B_list = []

# Iterate through each element in A with 0-based index
for i in range(N):
    num = A[i]
    pos = i + 1  # Convert to 1-based position
    
    # If the number has been seen before, set B_val to the last position
    if num in last_pos:
        B_val = last_pos[num]
    else:
        B_val = -1
    
    # Append the B value to the list
    B_list.append(B_val)
    
    # Update the last position of the number to the current position
    last_pos[num] = pos

# Output the B list separated by spaces
print(' '.join(map(str, B_list)))