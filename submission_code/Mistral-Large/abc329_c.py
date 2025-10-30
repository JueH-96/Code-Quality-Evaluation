import sys
from collections import defaultdict

def count_repetitions(N, S):
    # Dictionary to store the count of each character's consecutive sequences
    char_count = defaultdict(set)

    # Variables to keep track of the current character and its consecutive count
    current_char = ''
    current_count = 0

    # Iterate through the string
    for char in S:
        if char == current_char:
            current_count += 1
        else:
            if current_count > 0:
                char_count[current_char].add(current_count)
            current_char = char
            current_count = 1

    # Add the last sequence
    if current_count > 0:
        char_count[current_char].add(current_count)

    # Count the total number of unique repetitions
    total_repetitions = 0
    for counts in char_count.values():
        total_repetitions += sum(counts)

    return total_repetitions

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

# Calculate and print the result
result = count_repetitions(N, S)
print(result)