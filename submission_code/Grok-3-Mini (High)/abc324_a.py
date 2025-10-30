import sys
data = sys.stdin.read().split()
N = int(data[0])
A = list(map(int, data[1:N+1]))
if all(x == A[0] for x in A):
    print("Yes")
else:
    print("No")