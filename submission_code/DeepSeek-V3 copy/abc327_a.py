# Read input values
N = int(input())
S = input()

# Initialize a flag to check if any adjacent a and b are found
found = False

# Iterate through the string to check adjacent pairs
for i in range(N - 1):
    current_char = S[i]
    next_char = S[i + 1]
    if (current_char == 'a' and next_char == 'b') or (current_char == 'b' and next_char == 'a'):
        found = True
        break

# Print the result based on the flag
if found:
    print("Yes")
else:
    print("No")