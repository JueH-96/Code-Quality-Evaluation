import sys

# Read input from stdin
data = sys.stdin.read().strip()
B = int(data)

# Initialize answer to -1
answer = -1

# Loop through possible A values from 1 to 15
for A in range(1, 16):
    if A ** A == B:
        answer = A
        break  # No need to check further

# Output the answer
print(answer)