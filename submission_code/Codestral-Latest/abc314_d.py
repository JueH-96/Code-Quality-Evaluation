# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
S = list(data[index])
index += 1
Q = int(data[index])
index += 1

for _ in range(Q):
    t = int(data[index])
    index += 1
    x = int(data[index])
    index += 1
    c = data[index]
    index += 1

    if t == 1:
        S[x - 1] = c
    elif t == 2:
        S = [char.lower() for char in S]
    elif t == 3:
        S = [char.upper() for char in S]

print(''.join(S))