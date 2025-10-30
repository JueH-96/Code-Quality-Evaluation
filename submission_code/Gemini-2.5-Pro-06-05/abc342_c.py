import sys

# An expert solution for competitive programming should use fast I/O.
# sys.stdin.readline is generally faster than input().

# Read N, the length of the string. N is not directly used in the logic
# as Python strings handle their own length, but we must consume the line from input.
_ = int(sys.stdin.readline())
# Read the initial string S.
S = sys.stdin.readline().strip()
# Read Q, the number of operations.
Q = int(sys.stdin.readline())

# `from_chars` represents the original alphabet.
# `to_chars` will store what each character in `from_chars` transforms into.
# Initially, it's an identity mapping (e.g., 'a' maps to 'a').
from_chars = "abcdefghijklmnopqrstuvwxyz"
to_chars = list(from_chars)

# Process each of the Q replacement operations to build the final transformation map.
for _ in range(Q):
    c, d = sys.stdin.readline().split()
    
    # An optimization: if c and d are the same, the operation has no effect.
    if c == d:
        continue

    # Update the mapping: any original character that currently transforms
    # into `c` should now transform into `d`.
    for i in range(26):
        if to_chars[i] == c:
            to_chars[i] = d

# After all operations, we have the final transformation map.
# `str.maketrans()` creates a translation table that maps each character
# in `from_chars` to the character at the same position in `to_chars`.
translation_table = str.maketrans(from_chars, "".join(to_chars))

# Apply the single, cumulative transformation to the original string S
# using the highly optimized `str.translate()` method.
final_S = S.translate(translation_table)

# Print the final string to standard output.
print(final_S)