N = int(input())
A = list(map(int, input().split()))

# For each number i, find its middle occurrence
middle_positions = {}
count = {}

# Initialize count dictionary
for i in range(1, N+1):
    count[i] = 0

# Find middle occurrence for each number
for i in range(len(A)):
    num = A[i]
    count[num] += 1
    if count[num] == 2:  # Middle occurrence
        middle_positions[num] = i + 1  # 1-based index

# Create list of numbers with their middle positions
nums_with_pos = []
for i in range(1, N+1):
    nums_with_pos.append((i, middle_positions[i]))

# Sort by middle position
nums_with_pos.sort(key=lambda x: x[1])

# Print numbers in order
print(' '.join(str(x[0]) for x in nums_with_pos))