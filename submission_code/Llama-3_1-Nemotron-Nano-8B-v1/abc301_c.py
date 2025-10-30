# Read input
S = input().strip()
T = input().strip()

# Define the allowed characters
allowed = {'a', 't', 'c', 'o', 'd', 'e', 'r'}

# Count occurrences in S
s_counts = {c: 0 for c in allowed}
s_at = 0
for char in S:
    if char == '@':
        s_at += 1
    elif char in allowed:
        s_counts[char] += 1

# Count occurrences in T
t_counts = {c: 0 for c in allowed}
t_at = 0
for char in T:
    if char == '@':
        t_at += 1
    elif char in allowed:
        t_counts[char] += 1

# Calculate sum_low
sum_low = 0
for c in allowed:
    diff = s_counts[c] - t_counts[c]
    sum_low += max(0, diff)

# Check conditions
if sum_low <= t_at:
    print("Yes")
else:
    print("No")