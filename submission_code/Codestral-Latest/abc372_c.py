import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
S = list(data[2])
queries = [(int(data[i]), data[i + 1]) for i in range(3, 3 + 2 * Q, 2)]

# Function to count occurrences of "ABC" in the string
def count_abc(s):
    count = 0
    for i in range(len(s) - 2):
        if s[i] == 'A' and s[i + 1] == 'B' and s[i + 2] == 'C':
            count += 1
    return count

# Initialize the count of "ABC"
abc_count = count_abc(S)

# Process each query
for X, C in queries:
    X -= 1  # Convert to 0-based index
    if S[X] != C:
        # Update the count of "ABC" if the character changes
        if X < N - 2 and S[X] == 'A' and S[X + 1] == 'B' and S[X + 2] == 'C':
            abc_count -= 1
        if X > 0 and X < N - 1 and S[X - 1] == 'A' and S[X] == 'B' and S[X + 1] == 'C':
            abc_count -= 1
        if X > 1 and S[X - 2] == 'A' and S[X - 1] == 'B' and S[X] == 'C':
            abc_count -= 1

        # Update the character
        S[X] = C

        # Update the count of "ABC" after the change
        if X < N - 2 and S[X] == 'A' and S[X + 1] == 'B' and S[X + 2] == 'C':
            abc_count += 1
        if X > 0 and X < N - 1 and S[X - 1] == 'A' and S[X] == 'B' and S[X + 1] == 'C':
            abc_count += 1
        if X > 1 and S[X - 2] == 'A' and S[X - 1] == 'B' and S[X] == 'C':
            abc_count += 1

    print(abc_count)