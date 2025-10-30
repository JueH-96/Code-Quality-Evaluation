# YOUR CODE HERE

# Read the input string
s = input()

# Filter the string to keep only '2' characters
# We can use a list comprehension to create a list of '2's
# and then join them into a single string.
filtered_chars = [char for char in s if char == '2']
result_string = "".join(filtered_chars)

# Print the resulting string
print(result_string)