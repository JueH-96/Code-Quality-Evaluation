# Read the length of the string (N is not strictly used in this solution method, but part of input)
N = int(input())
# Read the string S
S = input()

# Find the index of the first occurrence of 'A'
pos_A = S.find('A')
# Find the index of the first occurrence of 'B'
pos_B = S.find('B')
# Find the index of the first occurrence of 'C'
pos_C = S.find('C')

# The condition (all A, B, C appeared) is met for the first time
# when the character at the largest of these first-occurrence indices is read.
# For example, if A is at index 0, C at index 1, and B at index 3 (S="ACXB..."),
# all three are seen once we process the character at index 3.
# The indices are 0-based.
latest_first_occurrence_index = max(pos_A, pos_B, pos_C)

# The number of characters checked to reach this index (inclusive) is index + 1.
# For example, if latest_first_occurrence_index is 3, characters S[0], S[1], S[2], S[3] are checked,
# which amounts to 3 + 1 = 4 characters.
result = latest_first_occurrence_index + 1

# Print the result
print(result)