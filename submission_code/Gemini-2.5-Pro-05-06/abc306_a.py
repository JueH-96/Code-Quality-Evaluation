# Read the integer N
N = int(input())
# Read the string S
S = input()

# Create a list of doubled characters
# For each character 'c' in S, 'c * 2' creates a string like "cc"
doubled_chars_list = [char * 2 for char in S]

# Join the list of doubled characters to form the final string
result_string = "".join(doubled_chars_list)

# Print the result string
print(result_string)