# Read the input string S from standard input
S = input()

# Use a list comprehension to iterate through the characters of S.
# For each character `char` in S, if `char` is '2', it's included in the list.
# This preserves the original order of the '2's.
list_of_twos = [char for char in S if char == '2']

# Join the characters in list_of_twos to form the final string.
# If list_of_twos is ['2', '2', '2'], "".join(list_of_twos) becomes "222".
result_string = "".join(list_of_twos)

# Print the resulting string to standard output
print(result_string)