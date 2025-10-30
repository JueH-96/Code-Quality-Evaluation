# YOUR CODE HERE
# Read input
N, T, P = map(int, input().split())
L = list(map(int, input().split()))

# Calculate days needed for each person to reach hair length T
days_needed = []
for length in L:
    if length >= T:
        days_needed.append(0)
    else:
        days_needed.append(T - length)

# Sort the days
days_needed.sort()

# The P-th person to reach T determines the answer
print(days_needed[P-1])