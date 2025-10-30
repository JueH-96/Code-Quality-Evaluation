# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
H = list(map(int, data[1:]))

T = 0
while sum(H) > 0:
    T += 1
    if T % 3 == 0:
        H[0] -= 3
    else:
        H[0] -= 1
    if H[0] <= 0:
        H.pop(0)

print(T)