# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
M = int(input[1])
H = list(map(int, input[2:]))

count = 0
for hands in H:
    if M >= hands:
        count += 1
        M -= hands
    else:
        break

print(count)