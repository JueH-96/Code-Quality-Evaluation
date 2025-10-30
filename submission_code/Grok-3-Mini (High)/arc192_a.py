import sys

# Read all input from stdin
data = sys.stdin.read().split()

# Read N
index = 0
N = int(data[index])
index += 1

# Read the array A
A = list(map(int, data[index:index + N]))
index += N

# Count the number of zeros in A
zero_count = 0
for a in A:
    if a == 0:
        zero_count += 1

# Determine if a good string exists
if zero_count != N or N % 4 == 0:
    print("Yes")
else:
    print("No")