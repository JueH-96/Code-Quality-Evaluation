# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]

count = 0
i = 0

while i < N:
    char = S[i]
    while i < N and S[i] == char:
        i += 1
    length = i - (i - 1)
    count += length

print(count)