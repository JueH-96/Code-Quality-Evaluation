# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
N = int(data[0])
S = list(data[1])
Q = int(data[2])

index = 3
for _ in range(Q):
    t = int(data[index])
    x = int(data[index + 1])
    c = data[index + 2]
    if t == 1:
        S[x - 1] = c
    elif t == 2:
        S = [char.lower() for char in S]
    elif t == 3:
        S = [char.upper() for char in S]
    index += 3

print("".join(S))