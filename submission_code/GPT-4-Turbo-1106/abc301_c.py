# Read the input strings
S = input().strip()
T = input().strip()

# Define the set of characters that can replace '@'
atcoder_chars = set('atcoder')

# Function to check if two characters match, considering the '@' replacement rule
def match(c1, c2):
    if c1 == c2:
        return True
    if c1 == '@' and c2 in atcoder_chars:
        return True
    if c2 == '@' and c1 in atcoder_chars:
        return True
    return False

# Check if it's possible to win by comparing each character in the strings
possible_to_win = all(match(s, t) for s, t in zip(S, T))

# Output the result
print("Yes" if possible_to_win else "No")