# YOUR CODE HERE
def first_occurrence_length(N, S):
    seen = {'A': 0, 'B': 0, 'C': 0}
    for i, char in enumerate(S):
        seen[char] += 1
        if all(seen.values()):
            return i + 1

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

# Print the result
print(first_occurrence_length(N, S))