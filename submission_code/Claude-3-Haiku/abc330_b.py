import sys

# Read input
line = sys.stdin.readline().split()
N = int(line[0])
L = int(line[1])
R = int(line[2])
A = [int(x) for x in sys.stdin.readline().split()]

# Solve the problem
result = []
for a in A:
    x = a
    if a < L:
        x = L
    elif a > R:
        x = R
    else:
        for i in range(L, R+1):
            if abs(i - a) <= abs(x - a):
                x = i
    result.append(x)

# Print the output
print(" ".join(map(str, result)))