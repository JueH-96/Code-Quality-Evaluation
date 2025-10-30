import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

B = [A[i] * A[i + 1] for i in range(N - 1)]

print(" ".join(map(str, B)))