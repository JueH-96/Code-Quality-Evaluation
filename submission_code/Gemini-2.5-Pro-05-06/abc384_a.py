# Read N, c1, c2 from the first line
line1_parts = input().split()
N = int(line1_parts[0])  # N is read as per spec, though not strictly used in list comprehension logic if S is trusted.
c1 = line1_parts[1]
c2 = line1_parts[2]

# Read the string S from the second line
S_original = input()

# Construct the new string using a list comprehension.
# For each character `char_s` in `S_original`:
#   If `char_s` is not equal to `c1`, it should be replaced by `c2`.
#   Otherwise (if `char_s` is equal to `c1`), it should remain `char_s`.
transformed_chars_list = [c2 if char_s != c1 else char_s for char_s in S_original]

# Join the characters in the list to form the final string
final_string = "".join(transformed_chars_list)

# Print the final string
print(final_string)