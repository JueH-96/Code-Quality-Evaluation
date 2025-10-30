import sys

# Read N, S, Q
# N is integer, S is string, Q is integer
N = int(sys.stdin.readline())
S = sys.stdin.readline().strip() # Read the string and remove trailing newline
Q = int(sys.stdin.readline())

# Initialize mapping for 26 lowercase letters 'a' through 'z'.
# mapping[i] will store the character that the original character chr(ord('a') + i) transforms into.
# Initially, each character maps to itself.
mapping = [chr(ord('a') + i) for i in range(26)]

# Process Q operations.
for _ in range(Q):
    # Read the operation pair (c_i, d_i).
    # c_i is the character to replace, d_i is the character to replace with.
    line = sys.stdin.readline().split()
    c = line[0]
    d = line[1]

    # If the replacement character is the same as the character to be replaced,
    # the operation has no effect on the mapping. We can skip updating.
    if c == d:
        continue

    # Update the mapping.
    # Iterate through all possible original characters (indexed 0 to 25).
    for k in range(26):
        # If the original character corresponding to index k (chr(ord('a')+k))
        # currently maps to character c (i.e., mapping[k] is c),
        # update its mapping so it now maps to d.
        if mapping[k] == c:
            mapping[k] = d

# After all operations, construct the final string by applying the final mapping
# to each character of the original string S.
result_chars = []
for char in S:
    # Get the index of the current character in the original string S.
    original_char_index = ord(char) - ord('a')
    # Look up its final mapped character from the mapping table.
    final_char = mapping[original_char_index]
    # Append the final character to our result list.
    result_chars.append(final_char)

# Join the list of characters into a single string.
final_string = "".join(result_chars)

# Print the final string.
print(final_string)