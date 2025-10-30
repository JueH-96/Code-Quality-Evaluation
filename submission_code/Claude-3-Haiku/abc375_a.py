# Read the input
N = int(input())
S = input()

# Initialize the count of valid seats
count = 0

# Iterate through the seats
for i in range(N-2):
    # Check if seats i and i+2 are occupied and seat i+1 is unoccupied
    if S[i] == '#' and S[i+1] == '.' and S[i+2] == '#':
        count += 1

# Print the answer
print(count)