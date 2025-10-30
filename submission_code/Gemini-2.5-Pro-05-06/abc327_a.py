# Read N, the length of the string.
N = int(input())
# Read the string S.
S = input()

# Check if "ab" or "ba" is present as a substring in S.
# The 'in' operator in Python is used for substring checking.
# If S contains "ab" or S contains "ba", then there are adjacent 'a' and 'b'.
if "ab" in S or "ba" in S:
    print("Yes")
else:
    print("No")