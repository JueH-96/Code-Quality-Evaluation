# YOUR CODE HERE
def is_1122_string(S):
    # Check if the length of S is even
    if len(S) % 2 != 0:
        return False

    # Check if each pair of characters is the same
    for i in range(0, len(S), 2):
        if S[i] != S[i + 1]:
            return False

    # Check if each character appears exactly zero or two times
    from collections import Counter
    char_count = Counter(S)
    for count in char_count.values():
        if count != 2:
            return False

    return True

# Read input
S = input().strip()

# Check if S is a 1122 string and print the result
if is_1122_string(S):
    print("Yes")
else:
    print("No")