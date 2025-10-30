# Read the length of the string N
N = int(input())

# Read the string S
S = input()

# Initialize a set to keep track of unique characters encountered
seen_chars = set()

# Iterate through the string character by character
# i represents the 0-based index of the current character
for i in range(N):
    # Add the current character to the set of seen characters
    seen_chars.add(S[i])

    # Check if all required characters ('A', 'B', 'C') are present in the set
    # Since the problem guarantees S only contains A, B, C,
    # and will eventually contain all three, we just need to check for their presence.
    if 'A' in seen_chars and 'B' in seen_chars and 'C' in seen_chars:
        # If all three are present, this is the first time the condition is met.
        # The number of characters checked is (i + 1) because i is a 0-based index.
        print(i + 1)
        # Once the condition is satisfied, we can stop and print the result,
        # as we are looking for the minimum length.
        break