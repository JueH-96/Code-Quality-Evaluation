# Read the input from stdin
a1, a2, a3, a4 = map(int, input().split())

# Count the frequency of each color
freq = [0] * 5
freq[a1] += 1
freq[a2] += 1
freq[a3] += 1
freq[a4] += 1

# Find the maximum number of operations that can be performed
max_operations = 0
for i in range(1, 5):
    if freq[i] >= 2:
        max_operations += freq[i] // 2

print(max_operations)