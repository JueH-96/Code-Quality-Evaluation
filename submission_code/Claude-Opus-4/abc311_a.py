# YOUR CODE HERE
n = int(input())
s = input()

# Set to keep track of which characters we've seen
seen = set()

# Check characters one by one from the left
for i in range(n):
    seen.add(s[i])
    
    # Check if all three characters have been seen
    if len(seen) == 3:  # or alternatively: if 'A' in seen and 'B' in seen and 'C' in seen
        print(i + 1)  # i is 0-indexed, so add 1 for the count
        break