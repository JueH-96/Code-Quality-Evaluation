# YOUR CODE HERE
N = int(input())
strings_list = []
for _ in range(N):
    current_string = input()
    strings_list.append(current_string)

# Sort the list of strings based on their length in ascending order.
# The `len` function is used as the key for sorting.
# Since string lengths are guaranteed to be distinct, there are no ties to break.
strings_sorted_by_length = sorted(strings_list, key=len)

# Concatenate the sorted strings into a single string.
# The `join` method is efficient for this. An empty string `""` is used as the separator,
# meaning the strings are joined directly one after another.
result_concatenated_string = "".join(strings_sorted_by_length)

# Print the final concatenated string to standard output.
print(result_concatenated_string)