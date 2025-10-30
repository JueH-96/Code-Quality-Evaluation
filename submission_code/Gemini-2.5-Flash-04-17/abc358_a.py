import sys

# Read the line from standard input
line = sys.stdin.readline().split()

# Extract the two strings S and T
S = line[0]
T = line[1]

# Check if S is "AtCoder" and T is "Land"
if S == "AtCoder" and T == "Land":
    print("Yes")
else:
    print("No")