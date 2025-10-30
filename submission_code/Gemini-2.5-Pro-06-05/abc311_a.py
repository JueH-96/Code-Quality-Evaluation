# Read the integer N from the first line of input.
# The value is not used in the logic, so we can assign it to a placeholder `_`.
_ = int(input())

# Read the string S from the second line of input.
S = input()

# Use a set to keep track of the unique characters ('A', 'B', 'C') encountered.
found_chars = set()

# Iterate through the string S, getting both the 0-based index `i` and the character `char`.
for i, char in enumerate(S):
    # Add the current character to the set. Sets automatically handle duplicates.
    found_chars.add(char)
    
    # Check if the set now contains all three distinct characters.
    if len(found_chars) == 3:
        # If it does, we have found our answer.
        # The problem asks for the number of characters checked,
        # which corresponds to the 1-based index (i + 1).
        print(i + 1)
        
        # Since we need the *first* time this condition is met,
        # we can stop the loop immediately.
        break