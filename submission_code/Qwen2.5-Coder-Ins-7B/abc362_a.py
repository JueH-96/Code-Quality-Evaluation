# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

R = int(data[0])
G = int(data[1])
B = int(data[2])
C = data[3]

if C == "Red":
    if R == 0:
        result = min(G, B)
    else:
        result = min(R, G, B)
elif C == "Green":
    if G == 0:
        result = min(R, B)
    else:
        result = min(R, G, B)
else:
    if B == 0:
        result = min(R, G)
    else:
        result = min(R, G, B)

print(result)