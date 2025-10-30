import sys

# Read the input string S
s = sys.stdin.readline().strip()

# This variable tracks the minimum index in the "ioio..." pattern (0-indexed)
# that is available to match the *next* character from S.
# Initially, the first character of S can potentially be matched starting from index 0.
current_pattern_idx = 0

# Iterate through each character in the input string S
for char in s:
    # Determine the character required at the current available pattern index 'current_pattern_idx'.
    # The target pattern is "ioioio..."
    # Index k requires 'i' if k is even (0, 2, 4, ...).
    # Index k requires 'o' if k is odd (1, 3, 5, ...).
    required_char_at_current_idx = 'i' if current_pattern_idx % 2 == 0 else 'o'

    # Check if the current character from S ('char') matches the character required
    # at the earliest available pattern index ('current_pattern_idx').
    if char == required_char_at_current_idx:
        # Yes, it matches. We successfully matched 'char' from S at index 'current_pattern_idx' in the pattern.
        # This character from S "uses up" the pattern position 'current_pattern_idx'.
        # The next character from S must therefore be matched at an index strictly greater than 'current_pattern_idx'.
        # The earliest available index for the next character is 'current_pattern_idx + 1'.
        current_pattern_idx = current_pattern_idx + 1
    else:
        # No, the current character 'char' from S does not match the character required
        # at the earliest available pattern index 'current_pattern_idx'.
        # We cannot use 'current_pattern_idx' for this character 'char' from S.
        # We must find the earliest index in the pattern, greater than 'current_pattern_idx',
        # that matches 'char'.
        # Since the pattern alternates ('io', 'oi', 'io', ...), the character required
        # at index 'current_pattern_idx + 1' is the opposite of the character required at
        # 'current_pattern_idx'.
        # Since 'char' didn't match the required character at 'current_pattern_idx',
        # it MUST match the required character at 'current_pattern_idx + 1'.
        # So, the earliest index >= current_pattern_idx where 'char' can be matched is 'current_pattern_idx + 1'.
        # After effectively matching 'char' at index 'current_pattern_idx + 1' in the pattern,
        # the next character from S must be matched starting from an index strictly greater than 'current_pattern_idx + 1'.
        # The earliest available index for the next character is '(current_pattern_idx + 1) + 1' = 'current_pattern_idx + 2'.
        current_pattern_idx = current_pattern_idx + 2

# After the loop finishes, 'current_pattern_idx' holds the minimum length of the
# prefix of the conceptual "ioio..." pattern that is required to accommodate all characters of S
# as a subsequence using the greedy earliest-match strategy. This value is the index *after*
# the last character of S is matched in the pattern. Thus, it represents the minimum number
# of pattern positions (0-indexed) that must be conceptually included.
required_pattern_length = current_pattern_idx

# The final target string must satisfy two conditions:
# 1. Its total length L must be even.
# 2. It must be long enough to contain the matched pattern prefix of length required_pattern_length.
#    This means the final length L must be >= required_pattern_length.
# We need the smallest length L such that L is even and L >= required_pattern_length.
# If required_pattern_length is even, the smallest such L is required_pattern_length itself.
# If required_pattern_length is odd, the smallest such L is required_pattern_length + 1.
# This calculation can be done concisely using integer division:
# The smallest even number >= X is ceil(X / 2) * 2.
# In integer arithmetic, ceil(X / 2) is equivalent to (X + 1) // 2 for non-negative X.
final_target_length = ((required_pattern_length + 1) // 2) * 2

# The number of characters inserted needed is the difference between the length of the
# final target string (which is the shortest valid target string that is a supersequence of S)
# and the length of the original string S.
insertions = final_target_length - len(s)

# Print the result
print(insertions)