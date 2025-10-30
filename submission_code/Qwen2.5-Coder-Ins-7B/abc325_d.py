# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
events = []

for i in range(N):
    T = int(data[2*i+1])
    D = int(data[2*i+2])
    events.append((T, D, 1))  # Product enters the range
    events.append((T + D, D, -1))  # Product leaves the range

events.sort()

current_time = 0
current_printing = 0
max_printing = 0

for T, D, event_type in events:
    current_time = T
    current_printing += event_type
    max_printing = max(max_printing, current_printing)

print(max_printing)