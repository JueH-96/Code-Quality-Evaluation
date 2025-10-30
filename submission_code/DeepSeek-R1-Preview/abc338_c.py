n = int(input())
Q = list(map(int, input().split()))
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Calculate x_max
x_max_candidates = []
for i in range(n):
    if A[i] > 0:
        x_max_candidates.append(Q[i] // A[i])
x_max = min(x_max_candidates) if x_max_candidates else 0

max_total = 0

for x in range(x_max + 1):
    # Check if x is valid for all B_i == 0
    valid = True
    for i in range(n):
        if B[i] == 0:
            if x * A[i] > Q[i]:
                valid = False
                break
    if not valid:
        continue
    
    # Calculate y candidates
    y_candidates = []
    for i in range(n):
        if B[i] == 0:
            continue
        remaining = Q[i] - x * A[i]
        if remaining < 0:
            y_i = -1
        else:
            y_i = remaining // B[i]
        y_candidates.append(y_i)
    
    # Determine y
    if not y_candidates:
        continue  # should not happen as per problem constraints
    min_y = min(y_candidates)
    if min_y < 0:
        y = 0
    else:
        y = min_y
    
    total = x + y
    if total > max_total:
        max_total = total

print(max_total)