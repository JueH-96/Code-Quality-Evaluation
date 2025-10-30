import sys

def solve(N):
    x = 0
    while 2**x <= N:
        y = 0
        while 3**y <= N//(2**x):
            if 2**x * 3**y == N:
                return 'Yes'
            y += 1
        x += 1
    return 'No'

# Read the inputs from stdin
N = int(sys.stdin.readline().strip())

# Solve the problem and write the answer to stdout
print(solve(N))