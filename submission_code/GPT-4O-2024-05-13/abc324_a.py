# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

if all(x == A[0] for x in A):
    print("Yes")
else:
    print("No")