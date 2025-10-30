import sys

# Read all input and split into a list
data = sys.stdin.read().split()

# Initialize index to 0
index = 0

# Read N and R
N = int(data[index])
index += 1
R = int(data[index])
index += 1

# Set initial rating
rating = R

# Process each of the N contests
for _ in range(N):
    D = int(data[index])
    index += 1
    A = int(data[index])
    index += 1
    
    # Check for rating update based on division and current rating
    if D == 1:  # ARC Div. 1
        if 1600 <= rating <= 2799:
            rating += A
    elif D == 2:  # ARC Div. 2
        if 1200 <= rating <= 2399:
            rating += A

# Output the final rating
print(rating)