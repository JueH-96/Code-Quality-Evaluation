import sys

# Read all input and split into a list
data = sys.stdin.read().split()

# Get N and the list of strings
N = int(data[0])
strs = data[1:]

# Flag to check if any palindrome is found
found = False

# Iterate through all pairs of indices
for i in range(N):
    for j in range(N):
        if i != j:  # Ensure i and j are distinct
            # Concatenate strings and check if palindrome
            concat_str = strs[i] + strs[j]
            if concat_str == concat_str[::-1]:  # Check if equal to its reverse
                found = True
                break  # Break inner loop
    if found:
        break  # Break outer loop

# Output the result
if found:
    print("Yes")
else:
    print("No")