import sys

# Read input from stdin
nums = list(map(int, sys.stdin.read().split()))

# Flag to check if any swap sorts the array
found = False

# Try swapping each pair of adjacent elements
for i in range(4):  # Possible swap positions: 0-1, 1-2, 2-3, 3-4
    # Create a copy of the list to avoid modifying the original
    temp = nums[:]
    # Perform the swap
    temp[i], temp[i + 1] = temp[i + 1], temp[i]
    # Check if the swapped list is sorted and equal to [1, 2, 3, 4, 5]
    if temp == [1, 2, 3, 4, 5]:
        found = True
        break  # No need to check further swaps

# Output the result
if found:
    print("Yes")
else:
    print("No")