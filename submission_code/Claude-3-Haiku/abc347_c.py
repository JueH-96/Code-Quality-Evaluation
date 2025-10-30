import sys

# Read input
N, A, B = map(int, input().split())
D = list(map(int, input().split()))

# Check if all plans can be scheduled on holidays
can_schedule = True
for i in range(N):
    day = (A + (D[i] - 1) % (A + B)) % (A + B)
    if day >= A:
        can_schedule = False
        break

# Print the answer
print("Yes" if can_schedule else "No")