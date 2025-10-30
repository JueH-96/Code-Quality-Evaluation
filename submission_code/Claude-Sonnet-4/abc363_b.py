# YOUR CODE HERE
N, T, P = map(int, input().split())
L = list(map(int, input().split()))

# Calculate how many days each person needs to reach hair length T
days_needed = []
for length in L:
    if length >= T:
        days_needed.append(0)  # Already at or above T
    else:
        days_needed.append(T - length)  # Days needed to reach T

# Sort the days needed
days_needed.sort()

# The answer is the number of days needed for the P-th person to reach T
# (P-th person in sorted order, so index P-1)
print(days_needed[P-1])