import sys

# Read all input from stdin and split into a list of strings
data = sys.stdin.read().split()

# Initialize index to keep track of position in data
index = 0

# Read N and A
N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))
index += N

# Read M and B
M = int(data[index])
index += 1
B = list(map(int, data[index:index + M]))
index += M

# Read L and C
L = int(data[index])
index += 1
C = list(map(int, data[index:index + L]))
index += L

# Read Q and X
Q = int(data[index])
index += 1
X = list(map(int, data[index:index + Q]))

# Compute all possible sums of one element from A, one from B, and one from C
possible_sums = {a + b + c for a in A for b in B for c in C}

# For each X_i, check if it is in possible_sums and print Yes or No
for x in X:
    if x in possible_sums:
        print("Yes")
    else:
        print("No")