def min_chars_to_check(s):
    seen = set()
    for i, char in enumerate(s):
        seen.add(char)
        if len(seen) == 3:  # We've seen all three characters
            return i + 1
    return len(s)  # This should not happen given the constraints

# Read input
N = int(input())
S = input()

# Output
print(min_chars_to_check(S))