# Read the input
n = int(input())
strings = [input() for _ in range(n)]

# Count the number of Takahashi strings
count = 0
for s in strings:
    if s == "Takahashi":
        count += 1

# Print the result
print(count)