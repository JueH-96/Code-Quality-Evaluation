# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Dictionary to store the most recent index of each element
last_seen = {}
B = []

for i in range(N):
    if A[i] in last_seen:
        B.append(last_seen[A[i]])
    else:
        B.append(-1)
    last_seen[A[i]] = i

print(" ".join(map(str, B)))