# YOUR CODE HERE
# Read the input string from standard input.
S = input()

# Use a generator expression to iterate through the string S.
# For each character `c` in S, it is included in the new sequence if it is not a vowel.
# The `"".join()` method then efficiently concatenates these characters into the final string.
# The vowels 'a', 'e', 'i', 'o', 'u' are specified in a string for the membership check.
result = "".join(c for c in S if c not in "aeiou")

# Print the final string to standard output.
print(result)