# YOUR CODE HERE
def first_all_abc(n, s):
    seen = set()
    for i in range(n):
        seen.add(s[i])
        if len(seen) == 3:  # When A, B, and C are all seen
            return i + 1  # Return the count of characters checked (1-indexed)
    return n  # Fallback, should not reach here due to problem constraints

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()
n = int(data[0])
s = data[1]

# Get the result and print it
result = first_all_abc(n, s)
print(result)