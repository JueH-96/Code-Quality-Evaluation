import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
A = list(map(int, data[index:index + N]))
is_geometric = True
if N > 2:
    for i in range(2, N):
        if A[i] * A[i-2] != A[i-1] ** 2:
            is_geometric = False
            break
if is_geometric:
    print("Yes")
else:
    print("No")