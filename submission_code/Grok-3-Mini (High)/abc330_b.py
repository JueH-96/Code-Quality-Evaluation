import sys
data = sys.stdin.read().split()
N = int(data[0])
L = int(data[1])
R = int(data[2])
A = list(map(int, data[3:3+N]))
result = [max(L, min(a, R)) for a in A]
print(' '.join(map(str, result)))