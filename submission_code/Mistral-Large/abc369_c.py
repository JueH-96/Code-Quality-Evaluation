import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Initialize variables
total_pairs = 0
current_length = 1

# Iterate through the sequence
for i in range(1, N):
    if A[i] - A[i-1] == A[i-1] - A[i-2]:
        current_length += 1
    else:
        total_pairs += (current_length * (current_length + 1)) // 2
        current_length = 1

# Add the last sequence
total_pairs += (current_length * (current_length + 1)) // 2

# Print the result
print(total_pairs)