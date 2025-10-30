import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()

# Read N and M
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Read the list of votes A
A = [int(x) for x in data[index:index + M]]

# Initialize count array, max_count, and winner
cnt = [0] * (N + 1)
max_count = 0
winner = 0  # This will be set after the first vote

# List to store the results
results = []

# Process each vote
for cand in A:
    # Increment the vote count for the candidate
    cnt[cand] += 1
    new_count = cnt[cand]
    
    # Update max_count and winner if necessary
    if new_count > max_count:
        max_count = new_count
        winner = cand
    elif new_count == max_count:
        if cand < winner:
            winner = cand
    
    # Append the current winner to results
    results.append(str(winner))

# Output the results, one per line
print('
'.join(results))