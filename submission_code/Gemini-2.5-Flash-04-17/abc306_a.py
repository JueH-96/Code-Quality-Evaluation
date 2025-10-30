import sys

# Read N from standard input
N = int(sys.stdin.readline())

# Read S from standard input
S = sys.stdin.readline().strip()

# Build the result string by repeating each character twice
# Use a generator expression and join for efficiency and conciseness
result = "".join(char * 2 for char in S)

# Print the result to standard output
print(result)