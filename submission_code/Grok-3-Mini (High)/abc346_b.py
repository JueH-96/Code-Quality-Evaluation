import sys
# Read input from stdin
data = sys.stdin.read().split()
W = int(data[0])
B = int(data[1])
L = W + B

# Define the period string
period_str = "wbwbwwbwbwbw"
P = 12

# Flag to check if such a substring exists
found = False

# Check for each possible starting offset modulo the period
for start_offset in range(P):
    num_w = 0
    # Compute the number of 'w's in the substring of length L starting at start_offset
    for i in range(L):
        pos = (start_offset + i) % P
        char = period_str[pos]
        if char == 'w':
            num_w += 1
    # If the number of 'w's equals W, set found to True and break
    if num_w == W:
        found = True
        break

# Output the result
if found:
    print("Yes")
else:
    print("No")