import sys

# Read input from stdin
data = sys.stdin.read().split()
L = int(data[0])
R = int(data[1])

# List to store the pairs (l, r)
pairs = []

# Current left endpoint
current_l = L

# Loop until current_l reaches R
while current_l < R:
    # Compute delta = R - current_l
    delta = R - current_l
    # Compute floor_log2_delta = floor(log2(delta))
    floor_log2_delta = delta.bit_length() - 1
    
    if current_l == 0:
        # If current_l is 0, it is divisible by all powers of two
        max_i = floor_log2_delta
    else:
        # Compute p = current_l & -current_l to find the largest power of two dividing current_l
        p = current_l & -current_l
        # Compute i_max_div = the exponent of p
        i_max_div = p.bit_length() - 1
        # max_i is the minimum of i_max_div and floor_log2_delta
        max_i = min(i_max_div, floor_log2_delta)
    
    # Compute the length of the good sequence
    length = 1 << max_i
    # Compute the right endpoint r
    r = current_l + length
    # Append the pair (current_l, r) to the list
    pairs.append((current_l, r))
    # Update current_l to r for the next interval
    current_l = r

# M is the number of pairs
M = len(pairs)

# Output M
print(M)

# Output each pair on a separate line
for pair in pairs:
    print(pair[0], pair[1])