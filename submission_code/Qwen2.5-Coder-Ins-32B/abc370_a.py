# YOUR CODE HERE
import sys

input = sys.stdin.read().strip().split()
L = int(input[0])
R = int(input[1])

if L == 1 and R == 0:
    print("Yes")
elif L == 0 and R == 1:
    print("No")
else:
    print("Invalid")