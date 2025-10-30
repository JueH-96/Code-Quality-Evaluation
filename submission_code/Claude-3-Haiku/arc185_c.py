import sys

# Read input
line1 = sys.stdin.readline().strip().split()
N = int(line1[0])
X = int(line1[1])
A = [int(x) for x in sys.stdin.readline().strip().split()]

# Solve the problem
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            if A[i] + A[j] + A[k] == X:
                print(i+1, j+1, k+1)
                sys.exit(0)

print(-1)