# Read input from stdin
N = int(input().strip())
S = input().strip()

# Find the first occurrence of "ABC"
position = S.find("ABC")

# If "ABC" is not found, find() returns -1
# Otherwise, it returns the index where "ABC" starts, which is 0-based
# We need to add 1 to convert it to 1-based as per the problem statement
print(position + 1 if position != -1 else -1)