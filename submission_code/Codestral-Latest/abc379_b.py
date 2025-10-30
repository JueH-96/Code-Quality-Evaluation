# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
S = data[2]

count = 0
max_count = 0

for i in range(N):
    if S[i] == 'O':
        count += 1
        if count == K:
            max_count += 1
            count = 0
    else:
        count = 0

print(max_count)