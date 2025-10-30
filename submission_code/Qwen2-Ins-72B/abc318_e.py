from collections import defaultdict

N = int(input())
A = list(map(int, input().split()))

# Initialize data structures
left_counts = defaultdict(int)
right_counts = defaultdict(int)
total_pairs = [0] * N

# Populate right_counts and total_pairs
for i, a in enumerate(A):
    right_counts[a] += 1
    total_pairs[i] = right_counts[a] - 1  # Subtract 1 to exclude the current element

# Populate left_counts and update total_pairs
for i, a in enumerate(A):
    right_counts[a] -= 1  # Exclude the current element from right_counts
    total_pairs[i] = total_pairs[i] * left_counts[a]
    left_counts[a] += 1

# Calculate the final answer
answer = sum(total_pairs)
print(answer)