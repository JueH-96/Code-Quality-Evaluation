# Read the input
N = int(input())
A = list(map(int, input().split()))

# Create a dictionary to store the indices of occurrences for each number
occurrences = {}
for i in range(1, N+1):
    occurrences[i] = []

# Record all occurrences (using 1-indexed positions as specified in the problem)
for idx, num in enumerate(A, start=1):
    occurrences[num].append(idx)

# Find the middle occurrence for each number
f_values = {}
for i in range(1, N+1):
    f_values[i] = occurrences[i][1]  # Second occurrence (1-indexed but 0-based in the list)

# Sort the numbers from 1 to N by their f(i) values
sorted_numbers = sorted(range(1, N+1), key=lambda x: f_values[x])

# Output the sorted sequence
print(" ".join(map(str, sorted_numbers)))