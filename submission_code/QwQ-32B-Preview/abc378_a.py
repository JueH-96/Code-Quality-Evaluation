# Read the input and convert to integers
A = list(map(int, input().split()))

# Initialize frequency list for colors 1 to 4
freq = [0] * 5

# Count the frequency of each color
for color in A:
    freq[color] += 1

# Calculate the number of pairs for each color and sum them up
pairs = sum(freq[c] // 2 for c in range(1, 5))

# Print the result
print(pairs)