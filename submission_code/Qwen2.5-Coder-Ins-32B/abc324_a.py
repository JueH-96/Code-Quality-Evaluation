# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
A = list(map(int, input[1:]))

if len(set(A)) == 1:
    print("Yes")
else:
    print("No")