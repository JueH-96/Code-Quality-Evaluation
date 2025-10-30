# N M
N, M = map(int, input().split())
# S_1 to S_N
strings = [input() for _ in range(N)]

# Function to calculate the Hamming distance between two strings
def hamming_distance(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

# Function to check if one string can be transformed into another by changing one character
def can_transform(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2)) == 1

# Create a list of all possible transformations for each string
transformations = []
for s in strings:
    transformations.append([s[:i] + c + s[i+1:] for i in range(M) for c in 'abcdefghijklmnopqrstuvwxyz' if s[:i] + c + s[i+1:] != s])

# Check if there exists a sequence that satisfies the condition
def check_sequence(seq):
    for i in range(N-1):
        if not can_transform(seq[i], seq[i+1]):
            return False
    return True

# Try all permutations of the strings
from itertools import permutations
for perm in permutations(strings):
    if check_sequence(perm):
        print('Yes')
        break
else:
    print('No')