import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
X = int(data[2])
A = list(map(int, data[3:3+N]))

A.insert(K, X)

print(" ".join(map(str, A)))