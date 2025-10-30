# Read N and M from the first line of input.
N, M = map(int, input().split())

# Read the sequence A of M integers from the second line.
# Constraints state 1 <= M <= N, so there will always be M elements to read.
A_elements = list(map(int, input().split()))

# Store A's elements in a set for efficient lookup.
A_set = set(A_elements)

# Initialize a list to store integers not found in A.
missing_numbers = []

# Iterate from 1 to N (inclusive).
for i in range(1, N + 1):
    # If i is not in A_set, it's a missing number.
    if i not in A_set:
        missing_numbers.append(i)
        # Since we iterate i in ascending order, missing_numbers will also be sorted.

# Output the count of missing numbers.
print(len(missing_numbers))

# Output the missing numbers themselves, space-separated.
# If missing_numbers is empty, print(*missing_numbers) will print a blank line,
# which matches the sample output format for C=0.
print(*missing_numbers)