# Read the first line of input
# It contains N, c_1, and c_2 separated by spaces.
# N is the length of the string S, but we can get it using len(S) if needed.
# We extract c_1 and c_2, which are the characters for replacement logic.
line1_parts = input().split()
# N = int(line1_parts[0]) # N is not strictly necessary for this problem's logic
c1 = line1_parts[1]
c2 = line1_parts[2]

# Read the second line, which is the string S
S = input()

# Initialize an empty list to store the characters of the modified string.
# Building a list of characters and then joining them is efficient in Python
# for string manipulations, as strings are immutable.
modified_chars = []

# Iterate through each character in the input string S
for char_s in S:
    # Check if the current character from S is equal to c_1
    if char_s == c1:
        # If it is c_1, keep the character as is
        modified_chars.append(char_s)
    else:
        # If it is not c_1, replace it with c_2
        modified_chars.append(c2)

# Join the list of characters to form the final modified string
result_string = "".join(modified_chars)

# Print the resulting string to standard output
print(result_string)