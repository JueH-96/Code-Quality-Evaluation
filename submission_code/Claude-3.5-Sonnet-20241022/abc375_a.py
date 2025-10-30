# Read input
N = int(input())
S = input()

# Initialize counter
count = 0

# Check each position from 1 to N-2
for i in range(N-2):
    # Check if current position and position+2 are occupied (#)
    # and position+1 is unoccupied (.)
    if S[i] == '#' and S[i+1] == '.' and S[i+2] == '#':
        count += 1

# Print result
print(count)