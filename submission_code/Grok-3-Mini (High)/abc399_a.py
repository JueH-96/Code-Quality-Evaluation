# Read input from stdin
n = int(input())
s = input()
t = input()

# Initialize distance counter
distance = 0

# Iterate through each position and count differences
for i in range(n):
    if s[i] != t[i]:
        distance += 1

# Output the Hamming distance
print(distance)