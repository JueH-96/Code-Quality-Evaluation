# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

S = data[0]
T = data[1]

result = []
j = 0

for i in range(len(S)):
    while T[j] != S[i]:
        j += 1
    result.append(j + 1)
    j += 1

print(" ".join(map(str, result)))