import sys

# Read all input and split into a list of strings
data = sys.stdin.read().split()

# Initialize index and read N
idx = 0
N = int(data[idx])
idx += 1

# Read intervals and store as (start, end)
intervals = []
for _ in range(N):
    T = int(data[idx])  # Read start time T
    D = int(data[idx + 1])  # Read duration D
    end_time = T + D  # Compute end time
    intervals.append((T, end_time))  # Store as (start, end)
    idx += 2  # Move index forward by 2 for next T and D

# Sort intervals by start time ascending
intervals.sort()

# Initialize count and last print time
count = 0
last_time = 0  # Initialize to 0 since T_i >= 1 and min separation is 1

# Iterate through each interval in sorted order
for start, end in intervals:
    possible_time = max(start, last_time + 1)  # Earliest time to print
    if possible_time <= end:  # Check if possible_time is within the interval
        count += 1  # Increment count if can print
        last_time = possible_time  # Update last print time

# Output the result
print(count)