# YOUR CODE HERE
import sys

# Read N and R
line1 = sys.stdin.readline().split()
n = int(line1[0])
r = int(line1[1])

# Initialize current rating
current_rating = r

# Loop through N contests
for _ in range(n):
    # Read division and performance
    line = sys.stdin.readline().split()
    d = int(line[0])
    a = int(line[1])

    # Check division and rating range for update
    if d == 1:
        # ARC Div 1: 1600 <= rating <= 2799
        if 1600 <= current_rating <= 2799:
            current_rating += a
    elif d == 2:
        # ARC Div 2: 1200 <= rating <= 2399
        if 1200 <= current_rating <= 2399:
            current_rating += a

# Print the final rating
print(current_rating)