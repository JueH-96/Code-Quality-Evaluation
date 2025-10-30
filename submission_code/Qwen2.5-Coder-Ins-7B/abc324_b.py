# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])

while N % 2 == 0:
    N //= 2

while N % 3 == 0:
    N //= 3

if N == 1:
    print("Yes")
else:
    print("No")