import sys
data = sys.stdin.read().split()
N = int(data[0])
K = int(data[1])
A = [int(x) for x in data[2:2+N]]
new_A = A[N - K:] + A[:N - K]
print(' '.join(map(str, new_A)))