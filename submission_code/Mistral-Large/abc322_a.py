import sys

def find_abc_position(N, S):
    # Iterate through the string to find the first occurrence of "ABC"
    for i in range(N - 2):
        if S[i:i+3] == "ABC":
            return i + 1  # Return the 1-based index
    return -1  # Return -1 if "ABC" is not found

# Read input from stdin
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]

# Find and print the position
result = find_abc_position(N, S)
print(result)