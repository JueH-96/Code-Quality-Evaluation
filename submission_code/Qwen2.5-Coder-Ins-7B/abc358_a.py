# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

S = data[0]
T = data[1]

if S == "AtCoder" and T == "Land":
    print("Yes")
else:
    print("No")