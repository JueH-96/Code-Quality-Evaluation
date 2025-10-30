# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
bases = []
for i in range(N):
    W, X = map(int, sys.stdin.readline().strip().split())
    bases.append((W, X))

bases.sort(key=lambda x: x[1])  # Sort bases by their UTC time

meeting_start, meeting_end = 9, 18
max_employees = 0
for i in range(N):
    W, X = bases[i]
    # Calculate the start and end times of the meeting at this base
    meeting_start_at_base = (meeting_start + X) % 24
    meeting_end_at_base = (meeting_end + X) % 24
    if meeting_start_at_base < meeting_end_at_base:
        # The meeting can be held at this base
        employees_at_base = min(W, meeting_end_at_base - meeting_start_at_base)
        max_employees += employees_at_base
    else:
        # The meeting cannot be held at this base
        employees_at_base = min(W, 24 - (meeting_start_at_base - meeting_end_at_base))
        max_employees += employees_at_base

print(max_employees)