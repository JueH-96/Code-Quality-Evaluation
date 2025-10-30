# Read input string S
S = input().strip()

# Process the string by removing the leftmost occurrences of "ABC"
while "ABC" in S:
    # Find the leftmost occurrence of "ABC"
    index = S.find("ABC")
    # Remove it from the string
    S = S[:index] + S[index+3:]

# Print the final string
print(S)