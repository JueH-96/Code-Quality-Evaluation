# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

count = 0
i = 0
while i < N:
    j = i + 1
    while j < N and S[j] == S[i]:
        j += 1
    length = j - i
    count += (length * (length + 1)) // 2
    i = j

print(count)