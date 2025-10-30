# Read the input sequence of 5 integers
a = list(map(int, input().split()))

# The target sorted sequence
sorted_a = [1, 2, 3, 4, 5]

# Flag to indicate if the sequence can be sorted by exactly one adjacent swap
can_sort_with_one_swap = False

# Iterate through all possible adjacent pairs to swap
# For a sequence of length 5, there are 4 adjacent pairs: (A_1, A_2), (A_2, A_3), (A_3, A_4), (A_4, A_5)
# These correspond to indices (0, 1), (1, 2), (2, 3), (3, 4)
for i in range(4):
    # Create a temporary copy of the list to perform the swap on
    temp_a = list(a)

    # Swap the adjacent elements at index i and i+1
    temp_a[i], temp_a[i+1] = temp_a[i+1], temp_a[i]

    # Check if the swapped list is now sorted
    if temp_a == sorted_a:
        # If it is sorted, we found a way to sort it with one adjacent swap
        can_sort_with_one_swap = True
        # We can stop checking as we only need to find one such operation
        break

# Print the result based on whether a sorting swap was found
if can_sort_with_one_swap:
    print("Yes")
else:
    print("No")