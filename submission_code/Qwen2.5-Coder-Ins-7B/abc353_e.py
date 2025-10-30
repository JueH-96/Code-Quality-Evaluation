# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1:]

def f(x, y):
    length = 0
    for i in range(min(len(x), len(y))):
        if x[i] == y[i]:
            length += 1
        else:
            break
    return length

total = 0
for i in range(N-1):
    for j in range(i+1, N):
        total += f(S[i], S[j])

print(total)