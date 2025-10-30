# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
H = list(map(int, data[2:]))

count = 0
for h in H:
    if M >= h:
        M -= h
        count += 1
    else:
        break

print(count)