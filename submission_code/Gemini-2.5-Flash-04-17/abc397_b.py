import sys

# Read the input string S from standard input.
S = sys.stdin.readline().strip()

# pattern_index tracks the current position we are trying to match in the infinite
# target pattern "ioioio..." (0-indexed).
# The target pattern is defined as:
# pattern[0] = 'i'
# pattern[1] = 'o'
# pattern[2] = 'i'
# pattern[3] = 'o'
# and so on.
# This means pattern[k] is 'i' if k is even, and 'o' if k is odd.
pattern_index = 0

# Iterate through each character in the input string S.
for char in S:
    # Determine the required character at the current pattern_index in the target pattern.
    # If pattern_index is even, the required character is 'i'.
    # If pattern_index is odd, the required character is 'o'.
    required_char = 'i' if pattern_index % 2 == 0 else 'o'

    # If the current character from S does not match the required pattern character
    # at the current pattern_index, it means the character at pattern_index
    # in the final string must be inserted.
    # We effectively 'insert' characters and advance the pattern_index until
    # the required character at the new pattern_index matches the current
    # character from S. Each increment of pattern_index inside this while loop
    # corresponds to one character being inserted.
    while char != required_char:
        # We move to the next position in the target pattern. This position
        # (pattern_index) will be filled by an inserted character.
        pattern_index += 1
        # Update the required character for the new pattern_index.
        required_char = 'i' if pattern_index % 2 == 0 else 'o'

    # Now, the current character from S matches the required character at pattern_index.
    # This character from S will be used to fill the position pattern_index in the
    # final string. We then advance pattern_index to the next position that the
    # subsequent characters from S might match.
    pattern_index += 1

# After processing all characters in S, pattern_index holds the total number of
# positions covered in the "ioioio..." pattern. This includes positions filled by
# characters from S and positions filled by necessary insertions made during the
# greedy matching process.
# This value, pattern_index, is the minimum length of a prefix of "ioioio..."
# that can contain S as a subsequence using the greedy matching strategy.
# Let this minimum prefix length be min_prefix_len = pattern_index.

# The problem requires the final resulting string to have an even length.
# The shortest valid final string must be a prefix of "ioioio..." with an even length,
# and its length must be at least min_prefix_len (because it must contain the
# subsequence matched greedily which requires covering up to pattern_index - 1).
# The smallest even length L that is greater than or equal to min_prefix_len is:
# - min_prefix_len itself, if min_prefix_len is even.
# - min_prefix_len + 1, if min_prefix_len is odd.
final_even_len = pattern_index
if final_even_len % 2 != 0:
    final_even_len += 1

# The total number of characters in the final required string is final_even_len.
# The number of original characters from S that are used in the final string is len(S).
# The number of inserted characters is the total characters in the final string
# minus the original characters from S.
num_insertions = final_even_len - len(S)

# Print the minimum number of characters that need to be inserted.
print(num_insertions)