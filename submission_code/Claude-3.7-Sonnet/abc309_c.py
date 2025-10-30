import sys
from collections import defaultdict

# Read input
N, K = map(int, input().split())

# Group events by day
day_to_change = defaultdict(int)
for _ in range(N):
    a, b = map(int, input().split())
    day_to_change[1] += b  # Start taking medicine on day 1
    day_to_change[a + 1] -= b  # Stop taking medicine on day a+1

# Sort days
days = sorted(day_to_change.keys())

# Simulate days and check the pill count
current_pill_count = 0

for day in days:
    # Update the pill count
    current_pill_count += day_to_change[day]
    
    # Check if the pill count is now less than or equal to K
    if current_pill_count <= K:
        print(day)
        sys.exit(0)

# If we've processed all events and still haven't found a day, 
# it means there's no day where pill count is <= K after the last event
# This should not happen based on the problem's constraints, but adding a safeguard
print(days[-1])