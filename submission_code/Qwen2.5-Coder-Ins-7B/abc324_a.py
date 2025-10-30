# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

if len(set(A)) == 1:
    print("Yes")
else:
    print("No")