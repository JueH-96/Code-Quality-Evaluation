# YOUR CODE HERE
n = int(input())
s = list(map(int, input().split()))

# Convert to set for O(1) lookup
s_set = set(s)

count = 0

# For each possible middle element B
for b in s:
    # For each possible first element A
    for a in s:
        if a < b:  # A must be less than B
            c = 2 * b - a  # Calculate C such that B - A = C - B
            if c in s_set and c > b:  # C must be in set and greater than B
                count += 1

print(count)