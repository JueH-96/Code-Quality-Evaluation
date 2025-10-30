# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
R = int(data[0])
G = int(data[1])
B = int(data[2])
C = data[3]

if C == "Red":
    print(min(G, B))
elif C == "Green":
    print(min(R, B))
elif C == "Blue":
    print(min(R, G))