import sys
from collections import defaultdict

# Read all input and split into a list
data = sys.stdin.read().split()

# Read N from the first element
N = int(data[0])

# Create a dictionary to store positions for each value
pos_dict = defaultdict(list)

# Read the array values and store positions (1-based)
for pos in range(1, N + 1):
    val = int(data[pos])
    pos_dict[val].append(pos)

# Initialize total count of triples
total = 0

# Iterate over each value's position list
for pos_list in pos_dict.values():
    M = len(pos_list)
    if M < 2:
        continue  # No pairs if less than 2 occurrences
    
    # Compute S_x for this value
    s_x = 0
    for k in range(M):
        pos_val = pos_list[k]  # Position value
        i = k + 1  # 1-based index in the list
        f_i = pos_val - i
        coeff = 2 * i - 1 - M
        s_x += f_i * coeff
    
    # Add S_x to total
    total += s_x

# Output the result
print(total)