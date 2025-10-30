N, A, B = map(int, input().split())
D = list(map(int, input().split()))

week_length = A + B

# Check if all plans can be scheduled on holidays
# For each plan day D_i, we need to check if there exists a starting day
# such that D_i mod (A+B) falls within first A days

# For each plan day, get its position in the week (0 to A+B-1)
positions = [d % week_length for d in D]

# Sort positions to check if they can all fit in first A days
positions.sort()

# Check if we can shift all positions to be within first A days
# by trying each possible starting day
possible = False
for start in range(week_length):
    valid = True
    for pos in positions:
        # Get position relative to start
        actual_pos = (pos - start) % week_length
        # Check if this position falls within holidays (first A days)
        if actual_pos >= A:
            valid = False
            break
    if valid:
        possible = True
        break

print("Yes" if possible else "No")