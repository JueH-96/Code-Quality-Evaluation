# YOUR CODE HERE
import sys

# Read N and M
line1 = sys.stdin.readline().split()
N = int(line1[0])
M = int(line1[1])

# Read A
line2 = sys.stdin.readline().split()
A_list = list(map(int, line2))

# Read B
line3 = sys.stdin.readline().split()
B_list = list(map(int, line3))

# Store elements of A in a set for efficient lookup
A_set = set(A_list)

# Combine A and B
combined_list = A_list + B_list

# Sort the combined list to get C
C = sorted(combined_list)

# Check for consecutive elements from A in C
found_consecutive_A = False
# Iterate up to the second to last element to check pairs (C[i], C[i+1])
for i in range(len(C) - 1):
    # Check if both the current element and the next element are in A_set
    if C[i] in A_set and C[i+1] in A_set:
        found_consecutive_A = True
        break # Found a pair, no need to check further

# Print the result
if found_consecutive_A:
    print("Yes")
else:
    print("No")
# YOUR CODE HERE