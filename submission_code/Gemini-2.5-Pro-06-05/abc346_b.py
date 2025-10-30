import sys

# Read the number of white (W) and black (B) keys from standard input.
# The map(int, ...) part converts the string inputs to integers.
W, B = map(int, sys.stdin.readline().split())

# The base pattern of the piano keyboard, which repeats infinitely.
# 'w' represents a white key, and 'b' represents a black key.
pattern = "wbwbwwbwbwbw"

# To find any possible substring, we construct a test string long enough
# to contain all relevant possibilities.
# The maximum length of the substring we need to find is W + B <= 200.
# Due to the periodicity of the keyboard pattern (period 12), we only need
# to check substrings that start at one of the first 12 positions.
# The longest such substring would start at index 11 and have a length of 200,
# requiring access up to index 211.
# Repeating the pattern 20 times (length 240) is more than sufficient.
s_test = pattern * 20

# The total length of the continuous segment we are looking for.
length_needed = W + B

# A flag to keep track of whether we've found a valid segment.
found = False

# We only need to check the 12 unique starting positions that arise from
# the repeating pattern of length 12.
for i in range(len(pattern)):
    # Extract a substring of the required length from our test string.
    substring = s_test[i : i + length_needed]
    
    # Check if the number of 'w's and 'b's in this substring
    # matches the required counts. We only need to check one count,
    # as the total length is fixed.
    if substring.count('w') == W and substring.count('b') == B:
        found = True
        # Once a valid segment is found, we can stop searching.
        break

# Print the final result based on whether a valid segment was found.
if found:
    print("Yes")
else:
    print("No")