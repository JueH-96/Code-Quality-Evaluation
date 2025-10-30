import sys

A = int(sys.stdin.readline().strip())

for B in range(1, 401):
    if A * B == 400:
        print(B)
        break
else:
    print(-1)