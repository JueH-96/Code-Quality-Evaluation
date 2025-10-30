# Read input
A = list(map(int, input().split()))

# Count frequency of each color
freq = {}
for color in A:
    freq[color] = freq.get(color, 0) + 1

# Count how many pairs we can make
operations = 0
for count in freq.values():
    operations += count // 2

print(operations)