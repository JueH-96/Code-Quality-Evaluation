import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
events = []

for i in range(1, N * 2 + 1, 2):
    T_i = int(data[i])
    D_i = int(data[i + 1])
    events.append((T_i, 1))  # Product enters
    events.append((T_i + D_i, -1))  # Product leaves

events.sort()

max_count = 0
current_count = 0
last_print_time = -2  # Initialize to a time before any event

for time, event_type in events:
    if event_type == 1:
        current_count += 1
    else:
        current_count -= 1

    if time > last_print_time + 1:
        if current_count > 0:
            max_count += 1
            last_print_time = time

print(max_count)