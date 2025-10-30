import sys

# Read the number of bean types
N = int(sys.stdin.readline())

# Use a dictionary to store the minimum deliciousness for each color.
# The keys will be colors (C_i), and the values will be the minimum deliciousness (A_i)
# found so far for that color.
# We initialize it as an empty dictionary.
min_deliciousness_by_color = {}

# Read and process each bean type
for _ in range(N):
    # Read the deliciousness A and color C for the current bean type
    A, C = map(int, sys.stdin.readline().split())

    # If this color is not yet a key in our dictionary, it means this is the first
    # bean of this color we've encountered. So, its deliciousness A is currently
    # the minimum for this color.
    if C not in min_deliciousness_by_color:
        min_deliciousness_by_color[C] = A
    else:
        # If this color is already in the dictionary, it means we've seen beans
        # of this color before. We update the stored minimum deliciousness for
        # this color if the current bean's deliciousness A is smaller than the
        # currently stored minimum.
        min_deliciousness_by_color[C] = min(min_deliciousness_by_color[C], A)

# After processing all N bean types, the 'min_deliciousness_by_color' dictionary
# holds the minimum deliciousness for every distinct color that appeared in the input.
# The problem asks us to choose a color to maximize the minimum possible deliciousness
# of the bean we eat. The minimum possible deliciousness for a chosen color is
# exactly the value stored for that color in our dictionary.
# Therefore, we need to find the maximum value among all the values stored in the
# 'min_deliciousness_by_color' dictionary.
# The .values() method returns a view object of the dictionary's values.
# The max() function finds the maximum element in this view.
result = max(min_deliciousness_by_color.values())

# Print the final result, which is the maximum of the minimum deliciousness values
# across all available colors.
print(result)